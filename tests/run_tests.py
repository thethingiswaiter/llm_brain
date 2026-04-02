import argparse
import unittest


QUICK_TEST_MODULES = [
    "tests.test_skill_loading",
    "tests.test_skill_routing",
    "tests.test_system_mcp_server",
    "tests.test_langchain_common_tools",
    "tests.test_core_reliability",
    "tests.test_failure_and_migration",
    "tests.test_memory_quality",
    "tests.test_tool_argument_prevalidation",
]

SLOW_INTEGRATION_TEST_MODULES = [
    "tests.test_cli_and_integration",
    "tests.test_mcp_manager_transport",
]


def build_suite(include_integration: bool) -> unittest.TestSuite:
    modules = list(QUICK_TEST_MODULES)
    if include_integration:
        modules.extend(SLOW_INTEGRATION_TEST_MODULES)
    return unittest.defaultTestLoader.loadTestsFromNames(modules)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run llm_brain test suites.")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Include slow integration suites (CLI + MCP transport).",
    )
    args = parser.parse_args()

    suite = build_suite(include_integration=args.all)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
