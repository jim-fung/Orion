#!/usr/bin/env python3
"""Validate the repository-local Markdown backlog without third-party packages.

Usage:
    python3 backlog/validate_backlog.py

The validator reads INDEX.md and every areas/*.md file beneath its own directory.
"""

from __future__ import print_function

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
INDEX_PATH = ROOT / "INDEX.md"
AREAS_PATH = ROOT / "areas"

ID_RE = re.compile(r"^ORION-[0-9]{4}$")
HEADING_RE = re.compile(r"^## (ORION-[0-9]+)\s+—\s+(.+?)\s*$")
FIELD_RE = re.compile(r"^\*\*(Status|Priority|Size|Dependencies):\*\*\s*(.*?)\s*$")
CHECKBOX_RE = re.compile(r"^- \[([ xX])\]\s+\*\*AC-[0-9]+\b")
INDEX_ROW_RE = re.compile(
    r"^\|\s*(ORION-[^|\s]+)\s*\|\s*"
    r"\[([^]]+)\]\(([^)#]+)#([^)]*)\)\s*\|\s*"
    r"([^|]+)\|\s*([^|]+)\|\s*([^|]+)\|\s*([^|]+)\|\s*([^|]+)\|\s*$"
)
STATUSES = {
    "proposed", "triaged", "ready", "in_progress", "blocked", "in_review",
    "done", "deferred", "rejected", "superseded",
}
PRIORITIES = {"P0", "P1", "P2", "P3"}
SIZES = {"XS", "S", "M", "L", "XL"}


class Problems(object):
    def __init__(self):
        self.items = []

    def add(self, path, line, message):
        try:
            relative = path.relative_to(ROOT.parent)
        except ValueError:
            relative = path
        self.items.append("{}:{}: {}".format(relative, line, message))


def read_lines(path, problems):
    try:
        return path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        problems.add(path, 1, "cannot read file: {}".format(error))
        return []


def parse_dependencies(value, path, line, problems):
    value = value.strip()
    if value in {"None", "—"}:
        return []
    dependencies = [part.strip() for part in re.split(r"[;,]", value) if part.strip()]
    if not dependencies:
        problems.add(path, line, "Dependencies must be None, —, or comma/semicolon-separated task IDs")
    for dependency in dependencies:
        if not ID_RE.fullmatch(dependency):
            problems.add(path, line, "invalid dependency ID {!r}".format(dependency))
    if len(dependencies) != len(set(dependencies)):
        problems.add(path, line, "dependency IDs must not be duplicated")
    return dependencies


def parse_index(problems):
    lines = read_lines(INDEX_PATH, problems)
    tasks = {}
    for number, line in enumerate(lines, 1):
        if not line.startswith("|") or line.startswith("| ID |"):
            continue
        match = INDEX_ROW_RE.fullmatch(line)
        if not match:
            if "ORION-" in line:
                problems.add(INDEX_PATH, number, "malformed task row")
            continue
        task_id, title, link_path, anchor, area, status, priority, size, blocked_by = (
            part.strip() for part in match.groups()
        )
        if not ID_RE.fullmatch(task_id):
            problems.add(INDEX_PATH, number, "invalid task ID {!r}".format(task_id))
        expected_anchor = "{}--{}".format(task_id.lower(), slug(title))
        if anchor != expected_anchor:
            problems.add(
                INDEX_PATH, number,
                "link anchor {!r} does not match {!r}".format(anchor, expected_anchor),
            )
        if status not in STATUSES:
            problems.add(INDEX_PATH, number, "invalid status {!r}".format(status))
        if priority not in PRIORITIES:
            problems.add(INDEX_PATH, number, "invalid priority {!r}".format(priority))
        if size not in SIZES:
            problems.add(INDEX_PATH, number, "invalid size {!r}".format(size))
        dependencies = parse_dependencies(blocked_by, INDEX_PATH, number, problems)
        if task_id in tasks:
            problems.add(INDEX_PATH, number, "duplicate task ID {}".format(task_id))
            continue
        tasks[task_id] = {
            "area": area,
            "dependencies": dependencies,
            "line": number,
            "link_path": link_path,
            "priority": priority,
            "size": size,
            "status": status,
            "title": title,
        }
    if not tasks:
        problems.add(INDEX_PATH, 1, "no task rows found")
    return tasks


def slug(title):
    value = title.lower()
    value = re.sub(r"[^a-z0-9 _-]", "", value)
    value = re.sub(r"[ _]+", "-", value).strip("-")
    return value


def section_end(lines, start):
    for index in range(start, len(lines)):
        if lines[index].startswith("## "):
            return index
    return len(lines)


def subsection(lines, start, end, heading):
    marker = "### " + heading
    for index in range(start, end):
        if lines[index].strip() == marker:
            next_index = index + 1
            while next_index < end and not lines[next_index].startswith("### "):
                next_index += 1
            return index, next_index
    return None


def evidence_section(lines, start, end):
    for heading in ("Completion evidence", "Evidence"):
        section = subsection(lines, start, end, heading)
        if section is not None:
            return section
    return None


def parse_area_file(path, problems):
    lines = read_lines(path, problems)
    tasks = {}
    headings = []
    for number, line in enumerate(lines, 1):
        match = HEADING_RE.fullmatch(line)
        if match:
            headings.append((number - 1, match.group(1), match.group(2)))
        elif line.startswith("## ORION-"):
            problems.add(path, number, "malformed task heading")

    for start, task_id, title in headings:
        end = section_end(lines, start + 1)
        if not ID_RE.fullmatch(task_id):
            problems.add(path, start + 1, "invalid task ID {!r}".format(task_id))
        if task_id in tasks:
            problems.add(path, start + 1, "duplicate task ID {}".format(task_id))
            continue
        fields = {}
        field_lines = {}
        for index in range(start + 1, end):
            match = FIELD_RE.fullmatch(lines[index])
            if match:
                name, value = match.groups()
                if name in fields:
                    problems.add(path, index + 1, "duplicate {} field".format(name))
                fields[name] = value.strip()
                field_lines[name] = index + 1
        for name in ("Status", "Priority", "Size", "Dependencies"):
            if name not in fields:
                problems.add(path, start + 1, "missing {} field".format(name))
        status = fields.get("Status", "")
        priority = fields.get("Priority", "").split()[0]
        size = fields.get("Size", "")
        dependencies = parse_dependencies(
            fields.get("Dependencies", ""), path, field_lines.get("Dependencies", start + 1), problems
        )
        if status and status not in STATUSES:
            problems.add(path, field_lines["Status"], "invalid status {!r}".format(status))
        if priority and priority not in PRIORITIES:
            problems.add(path, field_lines["Priority"], "invalid priority {!r}".format(priority))
        if size and size not in SIZES:
            problems.add(path, field_lines["Size"], "invalid size {!r}".format(size))
        tasks[task_id] = {
            "dependencies": dependencies,
            "end": end,
            "fields": fields,
            "line": start + 1,
            "lines": lines,
            "path": path,
            "priority": priority,
            "size": size,
            "start": start,
            "status": status,
            "title": title,
        }
    return tasks


def check_done_record(task, problems):
    lines = task["lines"]
    path = task["path"]
    acceptance = subsection(lines, task["start"], task["end"], "Acceptance criteria")
    if acceptance is None:
        problems.add(path, task["line"], "done task lacks an Acceptance criteria section")
    else:
        start, end = acceptance
        checkboxes = [
            (number + 1, match.group(1).lower())
            for number in range(start + 1, end)
            for match in [CHECKBOX_RE.match(lines[number])]
            if match
        ]
        if not checkboxes:
            problems.add(path, start + 1, "done task has no AC checkboxes")
        for line, state in checkboxes:
            if state != "x":
                problems.add(path, line, "done task has an unchecked acceptance criterion")

    evidence = evidence_section(lines, task["start"], task["end"])
    if evidence is None:
        problems.add(path, task["line"], "done task lacks a Completion evidence section")
        return
    start, end = evidence
    text = "\n".join(lines[start + 1:end])
    required = {
        "revision": r"\brevision\b",
        "date": r"\bdate\b",
        "command": r"\bcommand\b",
        "exit status": r"\bexit status\b",
        "PASS": r"\bPASS\b",
        "FAIL": r"\bFAIL\b",
        "SKIP": r"\bSKIP\b",
    }
    for label, pattern in required.items():
        if not re.search(pattern, text, re.IGNORECASE):
            problems.add(path, start + 1, "Completion evidence must record {}".format(label))


def check_relationships(index_tasks, area_tasks, problems):
    index_ids = set(index_tasks)
    area_ids = set(area_tasks)
    for task_id in sorted(index_ids - area_ids):
        problems.add(INDEX_PATH, index_tasks[task_id]["line"], "{} has no area task record".format(task_id))
    for task_id in sorted(area_ids - index_ids):
        task = area_tasks[task_id]
        problems.add(task["path"], task["line"], "{} is absent from INDEX.md".format(task_id))

    all_ids = index_ids | area_ids
    for source, tasks in (("INDEX.md", index_tasks), ("area records", area_tasks)):
        for task_id, task in tasks.items():
            line = task["line"]
            path = INDEX_PATH if source == "INDEX.md" else task["path"]
            for dependency in task["dependencies"]:
                if dependency not in all_ids:
                    problems.add(path, line, "{} depends on unknown task {}".format(task_id, dependency))
                if dependency == task_id:
                    problems.add(path, line, "{} cannot depend on itself".format(task_id))

    for task_id in sorted(index_ids & area_ids):
        index_task = index_tasks[task_id]
        area_task = area_tasks[task_id]
        expected_path = (ROOT / index_task["link_path"]).resolve()
        if expected_path != area_task["path"].resolve():
            problems.add(
                INDEX_PATH, index_task["line"],
                "{} links to {}, but its record is in {}".format(
                    task_id, index_task["link_path"], area_task["path"].relative_to(ROOT)
                ),
            )
        for field in ("status", "priority", "size", "dependencies"):
            if index_task[field] != area_task[field]:
                problems.add(
                    area_task["path"], area_task["line"],
                    "{} {} differs from INDEX.md".format(task_id, field),
                )
        if index_task["title"] != area_task["title"]:
            problems.add(area_task["path"], area_task["line"], "{} title differs from INDEX.md".format(task_id))

    for task_id, task in area_tasks.items():
        if task["status"] not in {"ready", "done"}:
            continue
        for dependency in task["dependencies"]:
            dependency_task = area_tasks.get(dependency)
            if dependency_task is not None and dependency_task["status"] != "done":
                problems.add(
                    task["path"], task["line"],
                    "{} is {} but blocking dependency {} is {}".format(
                        task_id, task["status"], dependency, dependency_task["status"]
                    ),
                )
        if task["status"] == "done":
            check_done_record(task, problems)


def main():
    problems = Problems()
    index_tasks = parse_index(problems)
    area_tasks = {}
    if not AREAS_PATH.is_dir():
        problems.add(AREAS_PATH, 1, "areas directory is missing")
    else:
        for path in sorted(AREAS_PATH.glob("*.md")):
            parsed = parse_area_file(path, problems)
            for task_id, task in parsed.items():
                if task_id in area_tasks:
                    problems.add(path, task["line"], "duplicate task ID {} across area files".format(task_id))
                else:
                    area_tasks[task_id] = task
    check_relationships(index_tasks, area_tasks, problems)
    if problems.items:
        print("Backlog validation failed ({} issue(s)):".format(len(problems.items)))
        for problem in problems.items:
            print("  " + problem)
        return 1
    print("Backlog validation passed: {} task(s) in INDEX.md and area records.".format(len(index_tasks)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
