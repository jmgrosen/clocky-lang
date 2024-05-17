#!/bin/bash

passed=0
failed=0

for f in tests/accept/*.cky; do
    if target/release/clocky typecheck "$f" >/dev/null 2>/dev/null; then
        echo "passed: $f"
        passed=$(( passed + 1))
    else
        echo "failed: $f"
        failed=$(( failed + 1))
    fi
done

for f in tests/reject/*.cky; do
    if target/release/clocky typecheck "$f" >/dev/null 2>/dev/null; then
        echo "failed: $f"
        failed=$(( failed + 1))
    else
        echo "passed: $f"
        passed=$(( passed + 1))
    fi
done

echo "$passed tests passed, $failed tests failed"
