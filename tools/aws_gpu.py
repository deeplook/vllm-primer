#!/usr/bin/env python3
"""Launch, inspect and delete a single vLLM GPU instance on AWS.

Credentials come only from ~/.aws (boto3's normal lookup). Everything this
tool creates is tagged Project=vllm-primer, and it only ever deletes things
carrying that tag. The instance has no inbound network access: you reach it
through an SSM Session Manager tunnel (`forward`), and it terminates itself
after VLLM_AUTO_TERMINATE_HOURS.

Commands: plan, policy, up, status, wait, logs, forward, down, purge.
Run `aws_gpu.py -h` or `aws_gpu.py <command> -h` for details.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

TAG_KEY = "Project"
TAG_VALUE = "vllm-primer"
ROLE_NAME = "vllm-primer-ssm"
SG_NAME = "vllm-primer"
HF_PARAM = "/vllm-primer/hf-token"
PORT = 8000
SSM_POLICY_ARN = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
POLICY_FILE = Path(__file__).with_name("aws_gpu_policy.json")
AMI_PARAM = (
    "/aws/service/deeplearning/ami/x86_64/"
    "base-oss-nvidia-driver-gpu-ubuntu-22.04/latest/ami-id"
)
# Service Quotas codes for "Running On-Demand G and VT" / "All G and VT Spot".
QUOTA_CODES = {
    "g": ("L-DB2E81BA", "L-3819A6DF"),
    "vt": ("L-DB2E81BA", "L-3819A6DF"),
    "p": ("L-417A185B", "L-7212CCBC"),
}
LIVE_STATES = ["pending", "running", "stopping", "stopped"]


class ToolError(Exception):
    """A problem to report to the user without a traceback."""


@dataclass(frozen=True)
class Config:
    instance_type: str = "g4dn.xlarge"
    serve_model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    image: str = "vllm/vllm-openai:v0.33.0"
    server_args: str = ""
    auto_terminate_hours: float = 4.0
    use_spot: bool = False
    volume_gb: int = 100
    instance_profile: str = ""
    has_hf_token: bool = False
    profile: str = ""
    region: str = ""

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> Config:
        def get(key: str, default: str) -> str:
            return env.get(key) or default

        return cls(
            instance_type=get("VLLM_INSTANCE_TYPE", cls.instance_type),
            serve_model=get("VLLM_SERVE_MODEL", cls.serve_model),
            image=get("VLLM_IMAGE", cls.image),
            server_args=get("VLLM_SERVER_ARGS", ""),
            auto_terminate_hours=float(get("VLLM_AUTO_TERMINATE_HOURS", "4")),
            use_spot=get("VLLM_USE_SPOT", "0") not in ("0", "false", "no"),
            volume_gb=int(get("VLLM_VOLUME_GB", "100")),
            instance_profile=get("VLLM_INSTANCE_PROFILE", ""),
            has_hf_token=bool(env.get("VLLM_HF_TOKEN")),
            profile=get("AWS_PROFILE", ""),
            region=get("AWS_REGION", ""),
        )


@dataclass
class Aws:
    """The boto3 clients the tool needs (fakes in tests)."""

    region: str
    ec2: Any
    ssm: Any
    iam: Any
    sts: Any
    pricing: Any
    quotas: Any


def make_aws(cfg: Config) -> Aws:
    import boto3
    from botocore.exceptions import NoCredentialsError, ProfileNotFound

    try:
        session = boto3.Session(
            profile_name=cfg.profile or None, region_name=cfg.region or None
        )
    except ProfileNotFound as err:
        raise ToolError(f"{err} (check AWS_PROFILE and ~/.aws/config)")
    if not session.region_name:
        raise ToolError("No region: set AWS_REGION or a region in ~/.aws/config")
    if session.get_credentials() is None:
        raise ToolError("No AWS credentials found in ~/.aws (try `aws configure`).")
    try:
        return Aws(
            region=session.region_name,
            ec2=session.client("ec2"),
            ssm=session.client("ssm"),
            iam=session.client("iam"),
            sts=session.client("sts"),
            pricing=session.client("pricing", region_name="us-east-1"),
            quotas=session.client("service-quotas"),
        )
    except NoCredentialsError as err:
        raise ToolError(str(err))


# --- instance bootstrap ----------------------------------------------------


def build_user_data(cfg: Config, region: str) -> str:
    """cloud-init script: self-destruct timer, then run the vLLM container."""
    minutes = max(1, round(cfg.auto_terminate_hours * 60))
    token = ""
    env_flag = ""
    if cfg.has_hf_token:
        token = (
            "set +x\n"
            f"HF_TOKEN=$(aws ssm get-parameter --region {region} --name {HF_PARAM} "
            "--with-decryption --query Parameter.Value --output text)\n"
            "set -x\n"
        )
        env_flag = '-e HF_TOKEN="$HF_TOKEN" '
    command = (
        f"--model {cfg.serve_model} --dtype half --max-model-len 4096 "
        "--enable-auto-tool-choice --tool-call-parser hermes"
    )
    if cfg.server_args:
        command += f" {cfg.server_args}"
    return (
        "#!/bin/bash\n"
        "set -euxo pipefail\n"
        "exec > >(tee -a /var/log/vllm-primer-setup.log) 2>&1\n"
        "# Instance-initiated shutdown is set to 'terminate', so this deletes it.\n"
        f'shutdown -h +{minutes} "vllm-primer auto-terminate"\n'
        f"{token}"
        "mkdir -p /opt/vllm-cache\n"
        "docker run -d --name vllm --restart unless-stopped --gpus all --ipc=host \\\n"
        f"  -p 127.0.0.1:{PORT}:{PORT} -v /opt/vllm-cache:/root/.cache/huggingface \\\n"
        f"  {env_flag}{cfg.image} {command}\n"
    )


def quota_codes(instance_type: str) -> tuple[str, str] | None:
    family = instance_type.split(".")[0]
    letters = "".join(c for c in family if c.isalpha())
    prefix = "vt" if letters.startswith("vt") else letters[:1]
    return QUOTA_CODES.get(prefix)


# --- read-only checks ------------------------------------------------------


@dataclass
class Plan:
    region: str
    account_arn: str
    instance_type: str
    vcpus: int
    gpu: str
    quota: float | None
    hourly_usd: float | None
    ami: str
    existing: list[str]
    problems: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    on_demand_usd: float | None = None  # only set when pricing Spot


def hourly_price(aws: Aws, instance_type: str) -> float | None:
    filters = {
        "instanceType": instance_type,
        "regionCode": aws.region,
        "operatingSystem": "Linux",
        "tenancy": "Shared",
        "preInstalledSw": "NA",
        "capacitystatus": "Used",
    }
    try:
        resp = aws.pricing.get_products(
            ServiceCode="AmazonEC2",
            Filters=[
                {"Type": "TERM_MATCH", "Field": k, "Value": v}
                for k, v in filters.items()
            ],
        )
        for item in resp.get("PriceList", []):
            terms = json.loads(item)["terms"]["OnDemand"]
            for term in terms.values():
                for dim in term["priceDimensions"].values():
                    return float(dim["pricePerUnit"]["USD"])
    except Exception:
        return None
    return None


def spot_price(aws: Aws, instance_type: str) -> float | None:
    """Cheapest current Spot price across availability zones."""
    try:
        history = aws.ec2.describe_spot_price_history(
            InstanceTypes=[instance_type],
            ProductDescriptions=["Linux/UNIX"],
            StartTime=datetime.now(UTC),
        )["SpotPriceHistory"]
        return min(float(h["SpotPrice"]) for h in history) if history else None
    except Exception:
        return None


def find_instances(
    aws: Aws, states: Sequence[str] = LIVE_STATES
) -> list[dict[str, Any]]:
    resp = aws.ec2.describe_instances(
        Filters=[
            {"Name": f"tag:{TAG_KEY}", "Values": [TAG_VALUE]},
            {"Name": "instance-state-name", "Values": list(states)},
        ]
    )
    return [i for r in resp["Reservations"] for i in r["Instances"]]


def make_plan(aws: Aws, cfg: Config) -> Plan:
    arn = aws.sts.get_caller_identity()["Arn"]
    plan = Plan(aws.region, arn, cfg.instance_type, 0, "?", None, None, "", [])
    if arn.endswith(":root"):
        plan.warnings.append(
            "You are using the account's ROOT user. Prefer an IAM user or SSO role."
        )

    try:
        info = aws.ec2.describe_instance_types(InstanceTypes=[cfg.instance_type])
        itype = info["InstanceTypes"][0]
        plan.vcpus = itype["VCpuInfo"]["DefaultVCpus"]
        gpus = itype.get("GpuInfo", {}).get("Gpus", [])
        if not gpus:
            plan.problems.append(f"{cfg.instance_type} has no GPU")
        else:
            mem = itype["GpuInfo"]["TotalGpuMemoryInMiB"] // 1024
            plan.gpu = f"{gpus[0]['Count']}x {gpus[0]['Name']} ({mem} GB)"
    except Exception as err:
        plan.problems.append(
            f"{cfg.instance_type} is not available in {aws.region}: {err}"
        )

    if codes := quota_codes(cfg.instance_type):
        code = codes[1] if cfg.use_spot else codes[0]
        try:
            quota = aws.quotas.get_service_quota(ServiceCode="ec2", QuotaCode=code)
            plan.quota = float(quota["Quota"]["Value"])
            if plan.vcpus and plan.quota < plan.vcpus:
                plan.problems.append(
                    f"vCPU quota is {plan.quota:g} but {cfg.instance_type} needs "
                    f"{plan.vcpus}. Request an increase in Service Quotas "
                    f"({'Spot' if cfg.use_spot else 'On-Demand'} {cfg.instance_type[0].upper()} instances)."
                )
        except Exception as err:
            plan.warnings.append(f"Could not read the vCPU quota: {err}")

    try:
        plan.ami = aws.ssm.get_parameter(Name=AMI_PARAM)["Parameter"]["Value"]
    except Exception as err:
        plan.problems.append(f"Could not resolve the Deep Learning AMI: {err}")

    plan.hourly_usd = hourly_price(aws, cfg.instance_type)
    if cfg.use_spot:
        plan.on_demand_usd = plan.hourly_usd
        plan.hourly_usd = spot_price(aws, cfg.instance_type)
    plan.existing = [
        i["InstanceId"] for i in find_instances(aws, ["pending", "running"])
    ]
    if plan.existing:
        plan.warnings.append(f"Already running: {', '.join(plan.existing)}")
    if cfg.has_hf_token and not cfg.instance_profile:
        plan.warnings.append(
            f"VLLM_HF_TOKEN will be stored as an encrypted SSM parameter ({HF_PARAM})."
        )
    return plan


def print_plan(plan: Plan, cfg: Config) -> None:
    cost = plan.hourly_usd
    cap = f"${cost * cfg.auto_terminate_hours:.2f}" if cost else "unknown"
    print(f"Account:   {plan.account_arn}")
    print(f"Region:    {plan.region}")
    print(f"Instance:  {plan.instance_type} - {plan.vcpus} vCPU, {plan.gpu}")
    print(f"Image:     {cfg.image} serving {cfg.serve_model}")
    print(f"AMI:       {plan.ami} (AWS Deep Learning Base, Ubuntu 22.04)")
    if cfg.use_spot:
        spot = f"${cost:.3f}/h now, can change" if cost else "price unknown"
        od = plan.on_demand_usd
        print(
            f"Pricing:   Spot, {spot} (On-Demand ${od:.3f}/h)"
            if od
            else f"Pricing:   Spot, {spot}"
        )
        print(
            "           Spot can be interrupted; an interrupted instance is terminated."
        )
    else:
        print(
            f"Pricing:   On-Demand, list price ${cost:.3f}/h"
            if cost
            else "Pricing:   On-Demand, price unknown"
        )
    print(f"Max spend: {cap} (auto-terminates after {cfg.auto_terminate_hours:g} h)")
    quota = "unknown" if plan.quota is None else f"{plan.quota:g} vCPU"
    print(f"Quota:     {quota}")
    print("Creates:   security group (no inbound rules), IAM role + instance profile")
    print(
        f"           for SSM, one {cfg.volume_gb} GB encrypted gp3 volume, 1 instance"
    )
    for warning in plan.warnings:
        print(f"WARNING: {warning}")
    for problem in plan.problems:
        print(f"PROBLEM: {problem}")


# --- up --------------------------------------------------------------------


def ensure_security_group(aws: Aws) -> str:
    vpcs = aws.ec2.describe_vpcs(Filters=[{"Name": "isDefault", "Values": ["true"]}])
    if not vpcs["Vpcs"]:
        raise ToolError(
            "No default VPC in this region; create one or use another region."
        )
    vpc_id = vpcs["Vpcs"][0]["VpcId"]
    found = aws.ec2.describe_security_groups(
        Filters=[
            {"Name": "group-name", "Values": [SG_NAME]},
            {"Name": "vpc-id", "Values": [vpc_id]},
        ]
    )["SecurityGroups"]
    if found:
        return str(found[0]["GroupId"])
    created = aws.ec2.create_security_group(
        GroupName=SG_NAME,
        Description="vllm-primer: no inbound, reach the instance through SSM",
        VpcId=vpc_id,
        TagSpecifications=[
            {
                "ResourceType": "security-group",
                "Tags": [{"Key": TAG_KEY, "Value": TAG_VALUE}],
            }
        ],
    )
    return str(created["GroupId"])


def ensure_instance_profile(aws: Aws, cfg: Config) -> tuple[str, bool]:
    """Return (profile name, created_now)."""
    if cfg.instance_profile:
        return cfg.instance_profile, False
    try:
        aws.iam.get_instance_profile(InstanceProfileName=ROLE_NAME)
        return ROLE_NAME, False
    except aws.iam.exceptions.NoSuchEntityException:
        pass
    trust = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"Service": "ec2.amazonaws.com"},
                "Action": "sts:AssumeRole",
            }
        ],
    }
    tags = [{"Key": TAG_KEY, "Value": TAG_VALUE}]
    aws.iam.create_role(
        RoleName=ROLE_NAME, AssumeRolePolicyDocument=json.dumps(trust), Tags=tags
    )
    aws.iam.attach_role_policy(RoleName=ROLE_NAME, PolicyArn=SSM_POLICY_ARN)
    aws.iam.create_instance_profile(InstanceProfileName=ROLE_NAME, Tags=tags)
    aws.iam.add_role_to_instance_profile(
        InstanceProfileName=ROLE_NAME, RoleName=ROLE_NAME
    )
    return ROLE_NAME, True


def put_hf_token(aws: Aws, cfg: Config, token: str) -> None:
    aws.ssm.put_parameter(
        Name=HF_PARAM,
        Value=token,
        Type="SecureString",
        Overwrite=True,
        Tags=[],
    )
    if cfg.instance_profile:
        return
    account = aws.sts.get_caller_identity()["Account"]
    policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": "ssm:GetParameter",
                "Resource": f"arn:aws:ssm:{aws.region}:{account}:parameter{HF_PARAM}",
            }
        ],
    }
    aws.iam.put_role_policy(
        RoleName=ROLE_NAME,
        PolicyName="read-hf-token",
        PolicyDocument=json.dumps(policy),
    )


def launch(aws: Aws, cfg: Config, profile: str, sg_id: str, ami: str) -> str:
    from botocore.exceptions import ClientError

    subnets = aws.ec2.describe_subnets(
        Filters=[{"Name": "default-for-az", "Values": ["true"]}]
    )["Subnets"]
    if not subnets:
        raise ToolError("No default subnets found.")
    images = aws.ec2.describe_images(ImageIds=[ami])["Images"]
    root = images[0]["RootDeviceName"] if images else "/dev/sda1"
    tags = [
        {"Key": TAG_KEY, "Value": TAG_VALUE},
        {"Key": "Name", "Value": "vllm-primer"},
    ]
    params: dict[str, Any] = {
        "ImageId": ami,
        "InstanceType": cfg.instance_type,
        "MinCount": 1,
        "MaxCount": 1,
        "IamInstanceProfile": {"Name": profile},
        "SecurityGroupIds": [sg_id],
        "UserData": build_user_data(cfg, aws.region),
        "InstanceInitiatedShutdownBehavior": "terminate",
        "MetadataOptions": {"HttpTokens": "required", "HttpEndpoint": "enabled"},
        "BlockDeviceMappings": [
            {
                "DeviceName": root,
                "Ebs": {
                    "VolumeSize": cfg.volume_gb,
                    "VolumeType": "gp3",
                    "Encrypted": True,
                    "DeleteOnTermination": True,
                },
            }
        ],
        "TagSpecifications": [
            {"ResourceType": "instance", "Tags": tags},
            {"ResourceType": "volume", "Tags": tags},
        ],
    }
    if cfg.use_spot:
        params["InstanceMarketOptions"] = {
            "MarketType": "spot",
            "SpotOptions": {
                "SpotInstanceType": "one-time",
                "InstanceInterruptionBehavior": "terminate",
            },
        }
    last: Exception | None = None
    for subnet in subnets:  # capacity is per AZ, so try each in turn
        for attempt in range(6):  # a fresh instance profile takes a while to be usable
            try:
                resp = aws.ec2.run_instances(SubnetId=subnet["SubnetId"], **params)
                return str(resp["Instances"][0]["InstanceId"])
            except ClientError as err:
                last = err
                code = err.response["Error"]["Code"]
                message = err.response["Error"]["Message"]
                if code == "InvalidParameterValue" and "Instance Profile" in message:
                    time.sleep(5)
                    continue
                if code in ("InsufficientInstanceCapacity", "Unsupported"):
                    break
                raise
    raise ToolError(f"Could not launch in any zone: {last}")


def cmd_up(
    aws: Aws, cfg: Config, args: argparse.Namespace, env: Mapping[str, str]
) -> int:
    plan = make_plan(aws, cfg)
    print_plan(plan, cfg)
    if plan.problems:
        return 1
    if not args.yes:
        print("\nNothing created. Re-run with --yes to launch (this starts billing).")
        return 0
    sg_id = ensure_security_group(aws)
    profile, created = ensure_instance_profile(aws, cfg)
    if token := env.get("VLLM_HF_TOKEN"):
        put_hf_token(aws, cfg, token)
    if created:
        time.sleep(10)
    instance_id = launch(aws, cfg, profile, sg_id, plan.ami)
    print(
        f"\nLaunched {instance_id}. The image download and model load take a few minutes."
    )
    print("Next: `make live-wait`, then `make live-forward` in another terminal.")
    return 0


# --- status / wait / logs / forward ---------------------------------------


def hours_up(instance: Mapping[str, Any]) -> float:
    launched: datetime = instance["LaunchTime"]
    return (datetime.now(UTC) - launched).total_seconds() / 3600


def cmd_status(
    aws: Aws, cfg: Config, args: argparse.Namespace, env: Mapping[str, str]
) -> int:
    instances = find_instances(aws)
    if not instances:
        print("No vllm-primer instances.")
        return 0
    price = hourly_price(aws, cfg.instance_type)
    info = aws.ssm.describe_instance_information()["InstanceInformationList"]
    ping = {i["InstanceId"]: i["PingStatus"] for i in info}
    for inst in instances:
        up = hours_up(inst)
        cost = f", ~${up * price:.2f} so far" if price else ""
        print(
            f"{inst['InstanceId']}  {inst['InstanceType']}  "
            f"{inst['State']['Name']}  {inst['Placement']['AvailabilityZone']}  "
            f"up {up:.1f} h{cost}  ssm: {ping.get(inst['InstanceId'], 'not registered')}"
        )
    return 0


def pick_instance(aws: Aws, explicit: str | None) -> str:
    if explicit:
        return explicit
    running = find_instances(aws, ["pending", "running"])
    if not running:
        raise ToolError(
            f"No running vllm-primer instance in {aws.region}. Did the launch run? "
            "`make live-up` only prints the plan; use `make live-up YES=1` to "
            "launch, or check `make live-status`."
        )
    if len(running) > 1:
        ids = ", ".join(i["InstanceId"] for i in running)
        raise ToolError(f"Several instances are running ({ids}); pass an instance id.")
    return str(running[0]["InstanceId"])


def run_shell(
    aws: Aws, instance_id: str, script: str, timeout: int = 60
) -> tuple[str, str]:
    """Run a shell command on the instance through SSM; return (status, output)."""
    sent = aws.ssm.send_command(
        InstanceIds=[instance_id],
        DocumentName="AWS-RunShellScript",
        Parameters={"commands": [script]},
    )
    command_id = sent["Command"]["CommandId"]
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        time.sleep(2)
        try:
            inv = aws.ssm.get_command_invocation(
                CommandId=command_id, InstanceId=instance_id
            )
        except aws.ssm.exceptions.InvocationDoesNotExist:
            continue
        if inv["Status"] not in ("Pending", "InProgress", "Delayed"):
            return str(inv["Status"]), str(inv.get("StandardOutputContent", ""))
    return "TimedOut", ""


def cmd_wait(
    aws: Aws, cfg: Config, args: argparse.Namespace, env: Mapping[str, str]
) -> int:
    instance_id = pick_instance(aws, args.instance)
    deadline = time.monotonic() + args.timeout
    print(f"Waiting for vLLM on {instance_id} (up to {args.timeout // 60} min)...")
    while time.monotonic() < deadline:
        info = aws.ssm.describe_instance_information(
            Filters=[{"Key": "InstanceIds", "Values": [instance_id]}]
        )["InstanceInformationList"]
        if info and info[0]["PingStatus"] == "Online":
            status, out = run_shell(
                aws, instance_id, f"curl -fsS localhost:{PORT}/health && echo READY"
            )
            if status == "Success" and "READY" in out:
                print("vLLM is healthy. Run `make live-forward` to tunnel it.")
                return 0
        print("  not ready yet")
        time.sleep(20)
    print("Timed out. Check `make live-logs`.", file=sys.stderr)
    return 1


def cmd_logs(
    aws: Aws, cfg: Config, args: argparse.Namespace, env: Mapping[str, str]
) -> int:
    instance_id = pick_instance(aws, args.instance)
    status, out = run_shell(
        aws,
        instance_id,
        "docker logs --tail 80 vllm 2>&1 || tail -n 40 /var/log/cloud-init-output.log",
    )
    print(out or f"(no output, status {status})")
    return 0


def forward_command(
    instance_id: str, region: str, profile: str, local_port: int
) -> list[str]:
    cmd = [
        "aws", "ssm", "start-session",
        "--target", instance_id,
        "--document-name", "AWS-StartPortForwardingSession",
        "--parameters", f"portNumber={PORT},localPortNumber={local_port}",
        "--region", region,
    ]  # fmt: skip
    return cmd + (["--profile", profile] if profile else [])


def cmd_forward(
    aws: Aws, cfg: Config, args: argparse.Namespace, env: Mapping[str, str]
) -> int:
    if shutil.which("aws") is None or shutil.which("session-manager-plugin") is None:
        raise ToolError(
            "`aws` and `session-manager-plugin` are required for the tunnel. "
            "Install: brew install awscli && brew install --cask session-manager-plugin"
        )
    instance_id = pick_instance(aws, args.instance)
    cmd = forward_command(instance_id, aws.region, cfg.profile, args.local_port)
    print(
        f"Tunnelling localhost:{args.local_port} -> {instance_id}:{PORT} (Ctrl-C stops)"
    )
    proc = subprocess.Popen(cmd)
    try:
        return proc.wait()
    except KeyboardInterrupt:
        # Ctrl-C goes to the whole foreground process group, so the AWS CLI is
        # already closing its session. Let it finish instead of dumping a
        # traceback; stopping the tunnel this way is normal use, not an error.
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.terminate()
            proc.wait(timeout=5)
        print("\nTunnel closed.")
        return 0


# --- down / purge ----------------------------------------------------------


def require_confirmation(env: Mapping[str, str], expected: str) -> None:
    if env.get("VLLM_PRIMER_CONFIRM") != expected:
        raise ToolError(
            f"Refusing: set VLLM_PRIMER_CONFIRM={expected} to confirm, e.g. "
            f"`VLLM_PRIMER_CONFIRM={expected} make clean-remote ...`"
        )


def cmd_down(
    aws: Aws, cfg: Config, args: argparse.Namespace, env: Mapping[str, str]
) -> int:
    require_confirmation(env, "terminate")
    tagged = {i["InstanceId"] for i in find_instances(aws, LIVE_STATES)}
    foreign = [i for i in args.instances if i not in tagged]
    if foreign:
        raise ToolError(
            f"Not live vllm-primer instances (wrong id or missing tag): {', '.join(foreign)}"
        )
    aws.ec2.terminate_instances(InstanceIds=list(args.instances))
    print(f"Terminating {', '.join(args.instances)}. Billing stops once they are gone.")
    return 0


def cmd_purge(
    aws: Aws, cfg: Config, args: argparse.Namespace, env: Mapping[str, str]
) -> int:
    require_confirmation(env, "purge")
    if live := find_instances(aws, LIVE_STATES + ["shutting-down"]):
        ids = ", ".join(i["InstanceId"] for i in live)
        raise ToolError(f"Instances still exist ({ids}); terminate them first.")

    def attempt(label: str, call: Any, retries: int = 1) -> None:
        for n in range(retries):
            try:
                call()
                print(f"deleted {label}")
                return
            except Exception as err:
                name = type(err).__name__
                if (
                    "NotFound" in name
                    or "NoSuchEntity" in name
                    or "ParameterNotFound" in name
                ):
                    return
                if n + 1 == retries:
                    print(f"could not delete {label}: {err}", file=sys.stderr)
                    return
                time.sleep(10)

    groups = aws.ec2.describe_security_groups(
        Filters=[
            {"Name": "group-name", "Values": [SG_NAME]},
            {"Name": f"tag:{TAG_KEY}", "Values": [TAG_VALUE]},
        ]
    )["SecurityGroups"]
    for group in groups:  # the network interface can outlive the instance briefly
        attempt(
            f"security group {group['GroupId']}",
            lambda g=group: aws.ec2.delete_security_group(GroupId=g["GroupId"]),
            retries=6,
        )
    tags = {}
    try:
        tags = {
            t["Key"]: t["Value"]
            for t in aws.iam.list_role_tags(RoleName=ROLE_NAME)["Tags"]
        }
    except Exception:
        pass
    if tags.get(TAG_KEY) == TAG_VALUE:
        attempt(
            "role policy read-hf-token",
            lambda: aws.iam.delete_role_policy(
                RoleName=ROLE_NAME, PolicyName="read-hf-token"
            ),
        )
        attempt(
            "role from instance profile",
            lambda: aws.iam.remove_role_from_instance_profile(
                InstanceProfileName=ROLE_NAME, RoleName=ROLE_NAME
            ),
        )
        attempt(
            "instance profile",
            lambda: aws.iam.delete_instance_profile(InstanceProfileName=ROLE_NAME),
        )
        attempt(
            "SSM policy attachment",
            lambda: aws.iam.detach_role_policy(
                RoleName=ROLE_NAME, PolicyArn=SSM_POLICY_ARN
            ),
        )
        attempt("role", lambda: aws.iam.delete_role(RoleName=ROLE_NAME))
    attempt("SSM parameter", lambda: aws.ssm.delete_parameter(Name=HF_PARAM))
    print("Purge finished.")
    return 0


def cmd_plan(
    aws: Aws, cfg: Config, args: argparse.Namespace, env: Mapping[str, str]
) -> int:
    plan = make_plan(aws, cfg)
    print_plan(plan, cfg)
    return 1 if plan.problems else 0


def cmd_policy(
    aws: Aws, cfg: Config, args: argparse.Namespace, env: Mapping[str, str]
) -> int:
    account = aws.sts.get_caller_identity()["Account"]
    text = POLICY_FILE.read_text()
    text = text.replace("<ACCOUNT_ID>", account).replace("<REGION>", aws.region)
    print(json.dumps(json.loads(text), indent=2))
    return 0


# --- CLI -------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aws_gpu.py",
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("plan", help="check access, quota and price; create nothing")
    sub.add_parser(
        "policy", help="print the least-privilege IAM policy for this account/region"
    )
    up = sub.add_parser("up", help="launch the instance (needs --yes)")
    up.add_argument("--yes", action="store_true", help="really create resources")
    status = sub.add_parser("status", help="list instances, uptime and cost so far")
    status.set_defaults(instance=None)
    for name, text in (
        ("wait", "wait until vLLM answers /health"),
        ("logs", "show the vLLM container logs"),
        ("forward", "tunnel localhost:PORT to the instance's vLLM"),
    ):
        p = sub.add_parser(name, help=text)
        p.add_argument(
            "instance", nargs="?", help="instance id (default: the only one)"
        )
        if name == "wait":
            p.add_argument("--timeout", type=int, default=1200, help="seconds")
        if name == "forward":
            p.add_argument("--local-port", type=int, default=PORT)
    down = sub.add_parser(
        "down", help="terminate named instances (needs VLLM_PRIMER_CONFIRM=terminate)"
    )
    down.add_argument("instances", nargs="+", metavar="INSTANCE_ID")
    sub.add_parser(
        "purge",
        help="delete the security group, role, and token (needs VLLM_PRIMER_CONFIRM=purge)",
    )
    return parser


HANDLERS = {
    "plan": cmd_plan,
    "policy": cmd_policy,
    "up": cmd_up,
    "status": cmd_status,
    "wait": cmd_wait,
    "logs": cmd_logs,
    "forward": cmd_forward,
    "down": cmd_down,
    "purge": cmd_purge,
}


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    env = dict(os.environ)
    cfg = Config.from_env(env)
    try:
        return HANDLERS[args.command](make_aws(cfg), cfg, args, env)
    except ToolError as err:
        print(f"error: {err}", file=sys.stderr)
        return 1
    except Exception as err:
        from botocore.exceptions import BotoCoreError, ClientError

        if isinstance(err, (ClientError, BotoCoreError)):
            print(f"AWS error: {err}", file=sys.stderr)
            return 1
        raise


if __name__ == "__main__":
    sys.exit(main())
