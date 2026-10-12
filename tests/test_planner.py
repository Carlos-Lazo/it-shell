import unittest
from semantic_search_engine.retrieval_service.query_analyzer import QueryAnalyzer
from semantic_search_engine.retrieval_service.planner import RulePlanner


class TestPlanner(unittest.TestCase):
    def setUp(self):
        self.analyzer = QueryAnalyzer()
        self.planner = RulePlanner()

    def test_single_step_query(self):
        analysis = self.analyzer.analyze("list files in current directory")
        self.assertFalse(analysis.is_complex)
        self.assertEqual(analysis.language, "en")

        plan = self.planner.plan(analysis)
        self.assertEqual(len(plan.steps), 1)
        self.assertEqual(plan.steps[0].id, "step_1")
        self.assertEqual(plan.steps[0].query, "list files in current directory")
        self.assertEqual(plan.steps[0].depends_on, ())

    def test_english_connectors(self):
        cases = [
            ("find logs and then compress them", ["find logs", "compress them"]),
            ("find logs then compress them", ["find logs", "compress them"]),
            ("find logs after that compress them", ["find logs", "compress them"]),
            ("find logs followed by compress them", ["find logs", "compress them"]),
            ("find logs next compress them", ["find logs", "compress them"]),
            ("find logs && compress them", ["find logs", "compress them"]),
            ("find logs | grep error", ["find logs", "grep error"]),
            ("find logs; compress them", ["find logs", "compress them"]),
        ]
        for query, expected_steps in cases:
            with self.subTest(query=query):
                analysis = self.analyzer.analyze(query)
                self.assertTrue(analysis.is_complex)
                plan = self.planner.plan(analysis)
                self.assertEqual([s.query for s in plan.steps], expected_steps)
                self.assertEqual(plan.steps[1].depends_on, ("step_1",))

    def test_spanish_connectors(self):
        cases = [
            ("buscar logs y luego comprimirlos", ["buscar logs", "comprimirlos"]),
            ("buscar logs después comprimirlos", ["buscar logs", "comprimirlos"]),
            ("buscar logs a continuación comprimirlos", ["buscar logs", "comprimirlos"]),
        ]
        for query, expected_steps in cases:
            with self.subTest(query=query):
                analysis = self.analyzer.analyze(query)
                self.assertTrue(analysis.is_complex)
                self.assertEqual(analysis.language, "es")
                plan = self.planner.plan(analysis)
                self.assertEqual([s.query for s in plan.steps], expected_steps)
                self.assertEqual(plan.steps[1].depends_on, ("step_1",))

    def test_multi_step_dependencies_chain(self):
        query = "find logs, then compress them, and after that send to server"
        analysis = self.analyzer.analyze(query)
        self.assertTrue(analysis.is_complex)

        plan = self.planner.plan(analysis)
        self.assertEqual(len(plan.steps), 3)
        self.assertEqual(plan.steps[0].id, "step_1")
        self.assertEqual(plan.steps[0].query, "find logs")
        self.assertEqual(plan.steps[0].depends_on, ())

        self.assertEqual(plan.steps[1].id, "step_2")
        self.assertEqual(plan.steps[1].query, "compress them")
        self.assertEqual(plan.steps[1].depends_on, ("step_1",))

        self.assertEqual(plan.steps[2].id, "step_3")
        self.assertEqual(plan.steps[2].query, "send to server")
        self.assertEqual(plan.steps[2].depends_on, ("step_2",))


if __name__ == "__main__":
    unittest.main()

