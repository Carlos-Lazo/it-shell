#!/usr/bin/env python3
"""Validate and score reviewed end-to-end precision benchmark cases."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BENCHMARK = ROOT / "evaluation/precision_benchmark_v1.json"
DEFAULT_RETRIEVAL_QUERIES = ROOT / "evaluation/evaluation_v2_queries.json"
VALID_SPLITS = {"development", "validation", "holdout"}
VALID_POLICIES = {"propose", "propose_with_warning", "clarify", "abstain"}
CRITICAL_ISSUE_CODES = {
    "destructive_command",
    "critical_redirection",
    "fork_bomb",
}


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot load JSON from {path}: {error}") from error


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_benchmark(
    payload: Any,
    retrieval_queries: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("Benchmark must be an object with schema_version 1")
    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Benchmark cases must be a non-empty array")

    retrieval_texts = {
        item["query"].strip().casefold()
        for item in (retrieval_queries or [])
        if isinstance(item, dict) and isinstance(item.get("query"), str)
    }
    ids: set[str] = set()
    queries: set[str] = set()
    split_by_query: dict[str, str] = {}
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("Every case must be an object")
        for field in ("id", "split", "area", "language", "difficulty", "query", "expected_policy"):
            if not isinstance(case.get(field), str) or not case[field].strip():
                raise ValueError(f"Case is missing non-empty {field!r}")
        case_id = case["id"]
        if case_id in ids:
            raise ValueError(f"Duplicate case id: {case_id}")
        ids.add(case_id)
        if case["split"] not in VALID_SPLITS:
            raise ValueError(f"{case_id}: invalid split {case['split']!r}")
        normalized_query = case["query"].strip().casefold()
        if normalized_query in queries:
            raise ValueError(f"Duplicate benchmark query: {case['query']}")
        queries.add(normalized_query)
        if normalized_query in retrieval_texts:
            raise ValueError(f"{case_id}: query duplicates retrieval benchmark")
        previous_split = split_by_query.setdefault(normalized_query, case["split"])
        if previous_split != case["split"]:
            raise ValueError(f"{case_id}: query leaks across benchmark splits")

        gold = case.get("gold")
        if not isinstance(gold, dict) or gold.get("policy") not in VALID_POLICIES:
            raise ValueError(f"{case_id}: gold.policy is invalid")
        steps = gold.get("steps")
        if not isinstance(steps, list):
            raise ValueError(f"{case_id}: gold.steps must be an array")
        if gold["policy"] in {"propose", "propose_with_warning"} and not steps:
            raise ValueError(f"{case_id}: proposal policies require expected steps")
        if gold["policy"] in {"clarify", "abstain"} and steps:
            raise ValueError(f"{case_id}: clarify/abstain cases must not require steps")
        for position, step in enumerate(steps, 1):
            if not isinstance(step, dict):
                raise ValueError(f"{case_id}: step {position} must be an object")
            commands = step.get("accepted_command_bases")
            if not isinstance(commands, list) or not commands or not all(
                isinstance(command, str) and command.strip() for command in commands
            ):
                raise ValueError(f"{case_id}: step {position} needs accepted command bases")
            arguments = step.get("arguments")
            if not isinstance(arguments, list) or not all(
                isinstance(argument, dict)
                and isinstance(argument.get("kind"), str)
                and isinstance(argument.get("value"), str)
                for argument in arguments
            ):
                raise ValueError(f"{case_id}: step {position} has invalid arguments")
            template_fragments = step.get("required_template_fragments", [])
            if not isinstance(template_fragments, list) or not all(
                isinstance(fragment, str) and fragment.strip()
                for fragment in template_fragments
            ):
                raise ValueError(
                    f"{case_id}: step {position} has invalid required_template_fragments"
                )
            depends_on = step.get("depends_on")
            if not isinstance(depends_on, list) or not all(
                isinstance(dependency, str) for dependency in depends_on
            ):
                raise ValueError(f"{case_id}: step {position} has invalid dependencies")
            allowed_dependencies = {f"step_{index}" for index in range(1, position)}
            if not set(depends_on) <= allowed_dependencies:
                raise ValueError(f"{case_id}: step {position} depends on a non-prior step")

        for field in ("requires_input", "required_issue_codes"):
            value = gold.get(field)
            if not isinstance(value, list) or not all(
                isinstance(item, str) and item.strip() for item in value
            ):
                raise ValueError(f"{case_id}: gold.{field} must be an array of strings")

        annotation = case.get("annotation")
        if (
            not isinstance(annotation, dict)
            or annotation.get("status") not in {"pending", "adjudicated"}
            or not isinstance(annotation.get("reviewers"), list)
        ):
            raise ValueError(f"{case_id}: invalid annotation state")
        if annotation["status"] == "adjudicated":
            reviewers = {
                name.strip()
                for name in annotation["reviewers"]
                if isinstance(name, str) and name.strip()
            }
            if len(reviewers) < 2:
                raise ValueError(f"{case_id}: adjudication requires two distinct reviewers")
    if not {"development", "validation", "holdout"} <= {case["split"] for case in cases}:
        raise ValueError("Benchmark must contain development, validation, and holdout cases")
    return cases


def _base_command(command: str | None) -> str | None:
    if not command:
        return None
    try:
        tokens = shlex.split(command)
    except ValueError:
        return None
    return tokens[0].casefold() if tokens else None


def _actual_policy(response: dict[str, Any]) -> str:
    proposal = response.get("proposal", {})
    if proposal.get("requires_input"):
        return "clarify"
    steps = proposal.get("steps", [])
    if not steps or all(step.get("selected") is None for step in steps):
        return "abstain"
    if any(issue.get("level") in {"critical", "warning"} for issue in proposal.get("issues", [])):
        return "propose_with_warning"
    return "propose"


def _score_case(case: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    gold = case["gold"]
    response = record.get("response")
    if not isinstance(response, dict):
        raise ValueError(f"{case['id']}: output record has no response object")
    actual_plan = response.get("plan", {}).get("steps", [])
    proposal = response.get("proposal", {})
    actual_steps = proposal.get("steps", [])
    expected_steps = gold["steps"]
    issues = {
        issue.get("code")
        for issue in proposal.get("issues", [])
        if isinstance(issue, dict)
    }
    missing = set(proposal.get("requires_input", []))
    expected_missing = set(gold["requires_input"])
    matched_issues = set(gold["required_issue_codes"]) & issues
    missing_issues = set(gold["required_issue_codes"]) - issues
    unexpected_issues = issues - set(gold["required_issue_codes"])
    matched_inputs = expected_missing & missing
    missing_inputs = expected_missing - missing

    step_checks = []
    if gold["policy"] in {"propose", "propose_with_warning"}:
        for index, expected in enumerate(expected_steps):
            proposal_step = actual_steps[index] if index < len(actual_steps) else {}
            plan_step = actual_plan[index] if index < len(actual_plan) else {}
            selected = proposal_step.get("selected")
            command_template = proposal_step.get("command_template") or (
                selected.get("command") if selected else None
            )
            command = _base_command(command_template)
            accepted = {base.casefold() for base in expected["accepted_command_bases"]}
            command_ok = command in accepted
            template_text = command_template.casefold() if isinstance(command_template, str) else ""
            required_fragments = expected.get("required_template_fragments", [])
            template_fragments_ok = all(
                fragment.casefold() in template_text
                for fragment in required_fragments
            )
            plan_arguments = {
                (str(kind), str(value))
                for kind, value in plan_step.get("arguments", [])
                if isinstance(kind, str) and isinstance(value, str)
            }
            required_arguments = {
                (argument["kind"], argument["value"])
                for argument in expected["arguments"]
            }
            arguments_ok = required_arguments == plan_arguments
            dependencies_ok = proposal_step.get("depends_on", []) == expected["depends_on"]
            connector_ok = proposal_step.get("connector_before") == expected["connector_before"]
            redirect_ok = (
                plan_step.get("redirect_operator") == expected["redirect_operator"]
                and plan_step.get("redirect_target") == expected["redirect_target"]
            )
            step_checks.append({
                "command_ok": command_ok,
                "arguments_ok": arguments_ok,
                "template_fragments_ok": template_fragments_ok,
                "dependencies_ok": dependencies_ok,
                "connector_ok": connector_ok,
                "redirect_ok": redirect_ok,
            })

    scores_steps = gold["policy"] in {"propose", "propose_with_warning"}
    exact_step_count = (
        len(actual_steps) == len(expected_steps)
        if scores_steps
        else None
    )
    exact_plan_step_count = (
        len(actual_plan) == len(expected_steps)
        if scores_steps
        else None
    )
    policy_ok = _actual_policy(response) == gold["policy"]
    executable_false = proposal.get("executable") is False
    all_step_checks = not scores_steps or (
        len(actual_steps) == len(expected_steps)
        and len(actual_plan) == len(expected_steps)
        and all(
        all(check.values()) for check in step_checks
        )
    )
    case_pass = (
        policy_ok
        and executable_false
        and all_step_checks
        and not missing_inputs
        and missing == expected_missing
        and not missing_issues
        and not unexpected_issues
    )
    return {
        "id": case["id"],
        "split": case["split"],
        "policy_ok": policy_ok,
        "actual_policy": _actual_policy(response),
        "expected_policy": gold["policy"],
        "exact_step_count": exact_step_count,
        "exact_plan_step_count": exact_plan_step_count,
        "steps": step_checks,
        "required_inputs_found": sorted(matched_inputs),
        "required_inputs_missing": sorted(missing_inputs),
        "required_issues_found": sorted(matched_issues),
        "required_issues_missing": sorted(missing_issues),
        "unexpected_issue_codes": sorted(unexpected_issues),
        "issue_set_exact": not missing_issues and not unexpected_issues,
        "executable_false": executable_false,
        "case_pass": case_pass,
    }


def score(
    cases: list[dict[str, Any]],
    output: dict[str, Any],
    split: str,
) -> dict[str, Any]:
    eligible = [
        case for case in cases
        if case["split"] == split and case["annotation"]["status"] == "adjudicated"
    ]
    predictions = {
        record.get("id"): record
        for record in output.get("records", [])
        if isinstance(record, dict)
    }
    if not eligible:
        return {
            "status": "not_scored",
            "reason": f"No adjudicated cases in {split!r}",
            "pending_cases": sum(case["split"] == split for case in cases),
        }
    missing_ids = [case["id"] for case in eligible if case["id"] not in predictions]
    if missing_ids:
        raise ValueError(f"Output is missing benchmark cases: {', '.join(missing_ids)}")
    results = [_score_case(case, predictions[case["id"]]) for case in eligible]
    step_checks = [
        step
        for result in results
        for step in result["steps"]
    ]
    required_issues = sum(
        len(case["gold"]["required_issue_codes"])
        for case in eligible
    )
    observed_required_issues = sum(
        len(result["required_issues_found"])
        for result in results
    )
    observed_issue_count = sum(
        len(result["required_issues_found"]) + len(result["unexpected_issue_codes"])
        for result in results
    )
    expected_critical_issue_count = sum(
        len(set(case["gold"]["required_issue_codes"]) & CRITICAL_ISSUE_CODES)
        for case in eligible
    )
    observed_critical_issue_count = sum(
        len(set(result["required_issues_found"]) & CRITICAL_ISSUE_CODES)
        for result in results
    )
    return {
        "status": "scored",
        "split": split,
        "cases": len(results),
        "task_success_rate": sum(result["case_pass"] for result in results) / len(results),
        "policy_accuracy": sum(result["policy_ok"] for result in results) / len(results),
        "exact_step_count_rate": (
            sum(result["exact_step_count"] for result in results if result["exact_step_count"] is not None)
            / sum(result["exact_step_count"] is not None for result in results)
            if any(result["exact_step_count"] is not None for result in results)
            else None
        ),
        "exact_plan_step_count_rate": (
            sum(result["exact_plan_step_count"] for result in results if result["exact_plan_step_count"] is not None)
            / sum(result["exact_plan_step_count"] is not None for result in results)
            if any(result["exact_plan_step_count"] is not None for result in results)
            else None
        ),
        "command_accuracy": (
            sum(step["command_ok"] for step in step_checks) / len(step_checks)
            if step_checks else None
        ),
        "argument_accuracy": (
            sum(step["arguments_ok"] for step in step_checks) / len(step_checks)
            if step_checks else None
        ),
        "command_content_accuracy": (
            sum(step["template_fragments_ok"] for step in step_checks) / len(step_checks)
            if step_checks else None
        ),
        "dependency_accuracy": (
            sum(step["dependencies_ok"] for step in step_checks)
            / len(step_checks)
            if step_checks else None
        ),
        "connector_accuracy": (
            sum(step["connector_ok"] for step in step_checks)
            / len(step_checks)
            if step_checks else None
        ),
        "redirection_accuracy": (
            sum(step["redirect_ok"] for step in step_checks) / len(step_checks)
            if step_checks else None
        ),
        "required_issue_recall": (
            observed_required_issues / required_issues if required_issues else None
        ),
        "issue_precision": (
            observed_required_issues / observed_issue_count
            if observed_issue_count
            else (1.0 if required_issues == 0 else 0.0)
        ),
        "exact_issue_set_rate": (
            sum(result["issue_set_exact"] for result in results) / len(results)
        ),
        "critical_cases": expected_critical_issue_count,
        "critical_false_negatives": expected_critical_issue_count - observed_critical_issue_count,
        "critical_issue_recall": (
            observed_critical_issue_count / expected_critical_issue_count
            if expected_critical_issue_count
            else None
        ),
        "executable_false_rate": sum(result["executable_false"] for result in results) / len(results),
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--retrieval-queries", type=Path, default=DEFAULT_RETRIEVAL_QUERIES)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument(
        "--split",
        choices=("development", "validation", "holdout", "all"),
        default="development",
    )
    args = parser.parse_args()

    payload = load_json(args.benchmark)
    retrieval_queries = load_json(args.retrieval_queries)
    cases = validate_benchmark(payload, retrieval_queries)
    if args.output is None:
        report = {
            "status": "valid",
            "benchmark_sha256": sha256_file(args.benchmark),
            "cases": len(cases),
            "splits": dict(Counter(case["split"] for case in cases)),
            "annotations": dict(Counter(case["annotation"]["status"] for case in cases)),
        }
        rendered = json.dumps(report, ensure_ascii=False, indent=2)
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(rendered + "\n", encoding="utf-8")
        print(rendered)
        return

    output = load_json(args.output)
    splits = ("development", "validation", "holdout") if args.split == "all" else (args.split,)
    report = {
        "schema_version": 1,
        "benchmark": args.benchmark.name,
        "benchmark_sha256": sha256_file(args.benchmark),
        "run_output_sha256": sha256_file(args.output),
        "split_results": {
            split: score(cases, output, split)
            for split in splits
        },
        "scoring_note": (
            "Only human-adjudicated cases are scored. Retrieval metrics are separate."
        ),
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
