"""Unit tests for tools/aws_gpu.py with fake AWS clients (nothing touches AWS)."""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import aws_gpu  # noqa: E402
from aws_gpu import Aws, Config, ToolError  # noqa: E402


class Fake:
    """A boto3-client stand-in: records calls, returns canned responses."""

    def __init__(self, **responses: Any) -> None:
        self.responses = responses
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.exceptions = SimpleNamespace(
            NoSuchEntityException=type("NoSuchEntityException", (Exception,), {}),
            InvocationDoesNotExist=type("InvocationDoesNotExist", (Exception,), {}),
        )

    def __getattr__(self, name: str) -> Callable[..., Any]:
        def call(**kwargs: Any) -> Any:
            self.calls.append((name, kwargs))
            result = self.responses.get(name, {})
            return result(**kwargs) if callable(result) else result

        return call

    def called(self, name: str) -> list[dict[str, Any]]:
        return [kw for n, kw in self.calls if n == name]


def instance(iid: str = "i-0abc", state: str = "running") -> dict[str, Any]:
    return {"InstanceId": iid, "State": {"Name": state}, "InstanceType": "g4dn.xlarge"}


def make_fake_aws(
    *,
    vcpus: int = 4,
    gpu: bool = True,
    quota: float = 4,
    arn: str = "arn:aws:iam::1:user/me",
    instances: list[dict[str, Any]] | None = None,
) -> Aws:
    itype: dict[str, Any] = {"VCpuInfo": {"DefaultVCpus": vcpus}}
    if gpu:
        itype["GpuInfo"] = {
            "Gpus": [{"Name": "T4", "Count": 1}],
            "TotalGpuMemoryInMiB": 16384,
        }
    return Aws(
        region="eu-central-1",
        ec2=Fake(
            describe_instance_types={"InstanceTypes": [itype]},
            describe_instances={"Reservations": [{"Instances": instances or []}]},
            describe_vpcs={"Vpcs": [{"VpcId": "vpc-1"}]},
            describe_security_groups={"SecurityGroups": [{"GroupId": "sg-1"}]},
            describe_subnets={"Subnets": [{"SubnetId": "subnet-1"}]},
            describe_images={"Images": [{"RootDeviceName": "/dev/sda1"}]},
            run_instances={"Instances": [{"InstanceId": "i-new"}]},
        ),
        ssm=Fake(get_parameter={"Parameter": {"Value": "ami-123"}}),
        iam=Fake(),
        sts=Fake(get_caller_identity={"Arn": arn, "Account": "1"}),
        pricing=Fake(),
        quotas=Fake(get_service_quota={"Quota": {"Value": quota}}),
    )


def ns(**kw: Any) -> argparse.Namespace:
    return argparse.Namespace(**kw)


def test_config_defaults_and_overrides() -> None:
    assert Config.from_env({}).instance_type == "g4dn.xlarge"
    assert not Config.from_env({}).use_spot
    cfg = Config.from_env(
        {"VLLM_INSTANCE_TYPE": "g5.xlarge", "VLLM_USE_SPOT": "1", "VLLM_HF_TOKEN": "x"}
    )
    assert cfg.instance_type == "g5.xlarge" and cfg.use_spot and cfg.has_hf_token


def test_ambient_hf_token_is_not_used() -> None:
    assert not Config.from_env({"HF_TOKEN": "hf_ambient"}).has_hf_token


def test_user_data_runs_vllm_privately_with_a_kill_switch() -> None:
    cfg = Config.from_env(
        {"VLLM_AUTO_TERMINATE_HOURS": "2", "VLLM_SERVER_ARGS": "--seed 1"}
    )
    script = aws_gpu.build_user_data(cfg, "eu-central-1")
    assert "shutdown -h +120" in script
    assert "-p 127.0.0.1:8000:8000" in script  # never exposed on the network
    assert "vllm/vllm-openai:v0.33.0 --model Qwen/Qwen2.5-0.5B-Instruct" in script
    assert "--dtype half" in script  # the T4 has no bf16
    assert "--tool-call-parser hermes" in script and "--seed 1" in script
    assert "HF_TOKEN" not in script


def test_user_data_fetches_the_token_from_ssm_not_inline() -> None:
    script = aws_gpu.build_user_data(
        Config.from_env({"VLLM_HF_TOKEN": "hf_secret_value"}), "eu-central-1"
    )
    assert "ssm get-parameter" in script and "--with-decryption" in script
    assert "hf_secret_value" not in script


@pytest.mark.parametrize(
    "itype,expected",
    [
        ("g4dn.xlarge", "L-DB2E81BA"),
        ("g6e.2xlarge", "L-DB2E81BA"),
        ("p4d.24xlarge", "L-417A185B"),
    ],
)
def test_quota_codes(itype: str, expected: str) -> None:
    codes = aws_gpu.quota_codes(itype)
    assert codes is not None and codes[0] == expected
    assert aws_gpu.quota_codes("m5.large") is None


def test_forward_command_uses_ssm_port_forwarding() -> None:
    cmd = aws_gpu.forward_command("i-1", "eu-central-1", "dev", 9000)
    assert cmd[:3] == ["aws", "ssm", "start-session"]
    assert "portNumber=8000,localPortNumber=9000" in cmd
    assert cmd[-2:] == ["--profile", "dev"]


def test_plan_ok_and_flags_root_user() -> None:
    plan = aws_gpu.make_plan(make_fake_aws(arn="arn:aws:iam::1:root"), Config())
    assert plan.problems == [] and plan.ami == "ami-123" and plan.vcpus == 4
    assert any("ROOT" in w for w in plan.warnings)


def test_plan_blocks_when_quota_is_too_low() -> None:
    plan = aws_gpu.make_plan(make_fake_aws(quota=0), Config())
    assert any("quota" in p for p in plan.problems)


def test_plan_blocks_non_gpu_types() -> None:
    plan = aws_gpu.make_plan(make_fake_aws(gpu=False), Config())
    assert any("no GPU" in p for p in plan.problems)


def test_up_without_yes_creates_nothing() -> None:
    aws = make_fake_aws()
    assert aws_gpu.cmd_up(aws, Config(), ns(yes=False), {}) == 0
    assert aws.ec2.called("run_instances") == []
    assert aws.iam.calls == []


def test_up_refuses_on_problems_even_with_yes() -> None:
    aws = make_fake_aws(quota=0)
    assert aws_gpu.cmd_up(aws, Config(), ns(yes=True), {}) == 1
    assert aws.ec2.called("run_instances") == []


def test_up_launches_a_locked_down_self_terminating_instance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(time, "sleep", lambda s: None)
    aws = make_fake_aws()
    aws.iam.responses["get_instance_profile"] = {}
    assert aws_gpu.cmd_up(aws, Config(), ns(yes=True), {}) == 0
    (call,) = aws.ec2.called("run_instances")
    assert call["InstanceInitiatedShutdownBehavior"] == "terminate"
    assert call["MetadataOptions"]["HttpTokens"] == "required"  # IMDSv2 only
    assert call["SecurityGroupIds"] == ["sg-1"]
    ebs = call["BlockDeviceMappings"][0]["Ebs"]
    assert ebs["Encrypted"] and ebs["DeleteOnTermination"]
    tags = call["TagSpecifications"][0]["Tags"]
    assert {"Key": "Project", "Value": "vllm-primer"} in tags
    assert "InstanceMarketOptions" not in call
    assert aws.ec2.called("authorize_security_group_ingress") == []  # no inbound rules


def test_spot_option_is_one_time_and_terminates() -> None:
    aws = make_fake_aws()
    aws_gpu.launch(aws, Config(use_spot=True), "prof", "sg-1", "ami-1")
    market = aws.ec2.called("run_instances")[0]["InstanceMarketOptions"]
    assert market["MarketType"] == "spot"
    assert market["SpotOptions"]["InstanceInterruptionBehavior"] == "terminate"


def test_down_requires_the_confirmation_variable() -> None:
    aws = make_fake_aws(instances=[instance()])
    with pytest.raises(ToolError, match="VLLM_PRIMER_CONFIRM=terminate"):
        aws_gpu.cmd_down(aws, Config(), ns(instances=["i-0abc"]), {})
    assert aws.ec2.called("terminate_instances") == []


def test_down_refuses_instances_without_our_tag() -> None:
    aws = make_fake_aws(instances=[instance("i-0abc")])
    env = {"VLLM_PRIMER_CONFIRM": "terminate"}
    with pytest.raises(ToolError, match="i-someone-else"):
        aws_gpu.cmd_down(aws, Config(), ns(instances=["i-someone-else"]), env)
    assert aws.ec2.called("terminate_instances") == []


def test_down_terminates_tagged_instances() -> None:
    aws = make_fake_aws(instances=[instance("i-0abc")])
    env = {"VLLM_PRIMER_CONFIRM": "terminate"}
    assert aws_gpu.cmd_down(aws, Config(), ns(instances=["i-0abc"]), env) == 0
    assert aws.ec2.called("terminate_instances") == [{"InstanceIds": ["i-0abc"]}]


def test_purge_needs_confirmation_and_no_live_instances() -> None:
    with pytest.raises(ToolError, match="VLLM_PRIMER_CONFIRM=purge"):
        aws_gpu.cmd_purge(make_fake_aws(), Config(), ns(), {})
    busy = make_fake_aws(instances=[instance()])
    with pytest.raises(ToolError, match="terminate them first"):
        aws_gpu.cmd_purge(busy, Config(), ns(), {"VLLM_PRIMER_CONFIRM": "purge"})
    assert busy.ec2.called("delete_security_group") == []


def test_purge_only_deletes_the_role_when_it_carries_our_tag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(time, "sleep", lambda s: None)
    aws = make_fake_aws()
    aws.iam.responses["list_role_tags"] = {
        "Tags": [{"Key": "Owner", "Value": "someone"}]
    }
    aws_gpu.cmd_purge(aws, Config(), ns(), {"VLLM_PRIMER_CONFIRM": "purge"})
    assert aws.iam.called("delete_role") == []


def test_main_reports_tool_errors_without_a_traceback(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(aws_gpu, "make_aws", lambda cfg: make_fake_aws())
    monkeypatch.delenv("VLLM_PRIMER_CONFIRM", raising=False)
    assert aws_gpu.main(["down", "i-0abc"]) == 1
    assert "VLLM_PRIMER_CONFIRM=terminate" in capsys.readouterr().err


def test_new_role_is_tagged_assumable_by_ec2_and_only_gets_ssm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(time, "sleep", lambda s: None)
    aws = make_fake_aws()

    def missing(**kwargs: Any) -> None:
        raise aws.iam.exceptions.NoSuchEntityException()

    aws.iam.responses["get_instance_profile"] = missing
    assert aws_gpu.cmd_up(aws, Config(), ns(yes=True), {}) == 0
    (role,) = aws.iam.called("create_role")
    assert {"Key": "Project", "Value": "vllm-primer"} in role["Tags"]
    assert "ec2.amazonaws.com" in role["AssumeRolePolicyDocument"]
    (attach,) = aws.iam.called("attach_role_policy")
    assert attach["PolicyArn"].endswith("AmazonSSMManagedInstanceCore")
    assert aws.iam.called("put_role_policy") == []  # no token, no extra access


def test_hf_token_goes_to_an_encrypted_parameter_with_read_only_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(time, "sleep", lambda s: None)
    aws = make_fake_aws()
    env = {"VLLM_HF_TOKEN": "hf_x"}
    assert aws_gpu.cmd_up(aws, Config.from_env(env), ns(yes=True), env) == 0
    (param,) = aws.ssm.called("put_parameter")
    assert param["Type"] == "SecureString" and param["Value"] == "hf_x"
    (policy,) = aws.iam.called("put_role_policy")
    assert "ssm:GetParameter" in policy["PolicyDocument"]
    assert "parameter/vllm-primer/hf-token" in policy["PolicyDocument"]
    (run,) = aws.ec2.called("run_instances")
    assert "hf_x" not in run["UserData"]


def test_spot_plan_uses_the_spot_price_not_the_on_demand_one() -> None:
    aws = make_fake_aws()
    aws.ec2.responses["describe_spot_price_history"] = {
        "SpotPriceHistory": [
            {"AvailabilityZone": "eu-central-1c", "SpotPrice": "0.300000"},
            {"AvailabilityZone": "eu-central-1a", "SpotPrice": "0.310600"},
        ]
    }
    aws.pricing.responses["get_products"] = {
        "PriceList": [
            '{"terms":{"OnDemand":{"a":{"priceDimensions":{"b":{"pricePerUnit":{"USD":"0.658"}}}}}}}'
        ]
    }
    plan = aws_gpu.make_plan(aws, Config(use_spot=True))
    assert plan.hourly_usd == 0.3 and plan.on_demand_usd == 0.658


def test_spot_plan_blocks_on_the_spot_quota_not_the_on_demand_one() -> None:
    aws = make_fake_aws()
    aws.quotas.responses["get_service_quota"] = lambda **kw: {
        "Quota": {"Value": 0 if kw["QuotaCode"] == "L-3819A6DF" else 4}
    }
    assert aws_gpu.make_plan(aws, Config(use_spot=False)).problems == []
    problems = aws_gpu.make_plan(aws, Config(use_spot=True)).problems
    assert any("Spot" in p for p in problems)
