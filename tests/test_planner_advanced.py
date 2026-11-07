import unittest

from semantic_search_engine.pipeline_builder_service.pipeline_builder import PipelineBuilder
from semantic_search_engine.pipeline_builder_service.validator import ProposalValidator
from semantic_search_engine.prompt_builder_service.prompt_builder import PromptBuilder
from semantic_search_engine.retrieval_service.planner import RulePlanner
from semantic_search_engine.retrieval_service.query_analyzer import QueryAnalyzer
from semantic_search_engine.shared.config import SelectionConfig
from semantic_search_engine.shared.contracts import CandidateCommand


class TestAdvancedPlannerArguments(unittest.TestCase):
    def setUp(self):
        self.planner = RulePlanner()
        self.analyzer = QueryAnalyzer()

    def test_extracts_path_filename_extension_and_size(self):
        plan = self.planner.plan(self.analyzer.analyze(
            "find files in /var/log named access.log larger than +100M"
        ))

        self.assertEqual(
            plan.steps[0].arguments,
            (("path", "/var/log"), ("filename", "access.log"), ("size", "+100M")),
        )

        extension_plan = self.planner.plan(
            self.analyzer.analyze("find archives matching *.tar.gz")
        )
        self.assertEqual(extension_plan.steps[0].arguments, (("extension", "*.tar.gz"),))

    def test_extracts_arguments_per_step_without_inheriting_them(self):
        plan = self.planner.plan(self.analyzer.analyze(
            "find files in /var/log then compress archive.tar.gz"
        ))

        self.assertEqual(plan.steps[0].arguments, (("path", "/var/log"),))
        self.assertEqual(plan.steps[1].arguments, (("filename", "archive.tar.gz"),))

    def test_does_not_extract_ambiguous_values_of_same_type(self):
        plan = self.planner.plan(self.analyzer.analyze("compare /var/log with /tmp/logs"))

        self.assertEqual(plan.steps[0].arguments, ())
        self.assertEqual(plan.steps[0].ambiguous_arguments, ("path",))

    def test_fills_only_matching_placeholders_and_quotes_values(self):
        builder = PipelineBuilder(SelectionConfig(
            min_score=0.45,
            alternative_margin=0.04,
            max_alternatives=3,
        ))
        candidate = CandidateCommand(
            "filesystem:1",
            "filesystem",
            1,
            "set output directory",
            "7z",
            "7z -o{Directory} <filename> {UTF-8|UTF-16LE}",
            0.8,
        )
        plan = self.planner.plan(self.analyzer.analyze(
            'extract archive to "/tmp/output folder"'
        ))

        proposal = builder.build(plan, {"step_1": [candidate]})

        self.assertEqual(
            proposal.steps[0].command_template,
            "7z -o'/tmp/output folder' <filename> {UTF-8|UTF-16LE}",
        )

    def test_leaves_non_matching_and_unprovided_placeholders_unchanged(self):
        builder = PipelineBuilder(SelectionConfig(
            min_score=0.45,
            alternative_margin=0.04,
            max_alternatives=3,
        ))
        candidate = CandidateCommand(
            "filesystem:1",
            "filesystem",
            1,
            "set output directory",
            "7z",
            "7z -o{Directory} -stx{Type}",
            0.8,
        )
        plan = self.planner.plan(self.analyzer.analyze("extract archive"))

        proposal = builder.build(plan, {"step_1": [candidate]})

        self.assertEqual(proposal.steps[0].command_template, "7z -o{Directory} -stx{Type}")

    def test_models_pipe_as_data_dependency(self):
        plan = self.planner.plan(self.analyzer.analyze(
            "find /var/log -type f | grep error"
        ))

        self.assertEqual([step.query for step in plan.steps], [
            "find /var/log -type f",
            "grep error",
        ])
        self.assertIsNone(plan.steps[0].connector_before)
        self.assertEqual(plan.steps[1].connector_before, "|")
        self.assertEqual(plan.steps[1].depends_on, ("step_1",))

    def test_models_conditional_and_sequential_connectors(self):
        plan = self.planner.plan(self.analyzer.analyze(
            "find logs && compress logs; send archive"
        ))

        self.assertEqual([step.connector_before for step in plan.steps], [None, "&&", ";"])
        self.assertEqual([step.depends_on for step in plan.steps], [
            (),
            ("step_1",),
            ("step_2",),
        ])

    def test_models_output_and_append_redirections(self):
        plan = self.planner.plan(self.analyzer.analyze(
            'find /var/log > "errors report.txt" && grep error >> results.txt'
        ))

        self.assertEqual([step.query for step in plan.steps], [
            "find /var/log",
            "grep error",
        ])
        self.assertEqual(plan.steps[0].redirect_operator, ">")
        self.assertEqual(plan.steps[0].redirect_target, "errors report.txt")
        self.assertEqual(plan.steps[1].redirect_operator, ">>")
        self.assertEqual(plan.steps[1].redirect_target, "results.txt")
        self.assertEqual(plan.steps[1].connector_before, "&&")

    def test_extracts_natural_language_append_redirection(self):
        plan = self.planner.plan(self.analyzer.analyze(
            "list files in /tmp and append the output to /tmp/listing.txt"
        ))

        self.assertEqual(len(plan.steps), 1)
        self.assertEqual(plan.steps[0].query, "list files in /tmp")
        self.assertEqual(plan.steps[0].arguments, (("path", "/tmp"),))
        self.assertEqual(plan.steps[0].ambiguous_arguments, ())
        self.assertEqual(plan.steps[0].redirect_operator, ">>")
        self.assertEqual(plan.steps[0].redirect_target, "/tmp/listing.txt")

    def test_models_three_step_pipe_redirection_and_condition(self):
        plan = self.planner.plan(self.analyzer.analyze(
            "find /var/log | grep error > errors.txt && sort errors.txt"
        ))

        self.assertEqual([step.query for step in plan.steps], [
            "find /var/log",
            "grep error",
            "sort errors.txt",
        ])
        self.assertEqual([step.depends_on for step in plan.steps], [
            (),
            ("step_1",),
            ("step_2",),
        ])
        self.assertEqual([step.connector_before for step in plan.steps], [
            None,
            "|",
            "&&",
        ])
        self.assertEqual(plan.steps[1].redirect_operator, ">")
        self.assertEqual(plan.steps[1].redirect_target, "errors.txt")

    def test_ignores_shell_operators_inside_quoted_text(self):
        plan = self.planner.plan(self.analyzer.analyze(
            'search for "a | b > c" in /var/log'
        ))

        self.assertEqual(len(plan.steps), 1)
        self.assertIsNone(plan.steps[0].redirect_operator)
        self.assertEqual(plan.steps[0].arguments, (("path", "/var/log"),))

    def test_requests_missing_file_and_destination_for_vague_compression(self):
        builder = PipelineBuilder(SelectionConfig(
            min_score=0.45,
            alternative_margin=0.04,
            max_alternatives=3,
        ))
        candidate = CandidateCommand(
            "filesystem:1", "filesystem", 1, "compress file", "gzip", "gzip <filename>", 0.8
        )
        plan = self.planner.plan(self.analyzer.analyze("compress file"))
        proposal = builder.build(plan, {"step_1": [candidate]})

        validated = ProposalValidator().validate(proposal)

        self.assertEqual(validated.requires_input, ("destination", "filename"))
        self.assertIn("missing_input", [issue.code for issue in validated.issues])
        self.assertFalse(validated.executable)

    def test_explicit_redirection_satisfies_compression_destination(self):
        candidate = CandidateCommand(
            "filesystem:1",
            "filesystem",
            1,
            "compress file",
            "gzip",
            "gzip <filename> {destination}",
            0.8,
        )
        plan = self.planner.plan(self.analyzer.analyze(
            "compress file > archive.gz"
        ))
        proposal = PipelineBuilder(SelectionConfig(
            min_score=0.45,
            alternative_margin=0.04,
            max_alternatives=3,
        )).build(plan, {"step_1": [candidate]})

        validated = ProposalValidator().validate(proposal)

        self.assertEqual(validated.requires_input, ("filename",))
        self.assertEqual(validated.steps[0].redirect_operator, ">")
        self.assertEqual(validated.steps[0].redirect_target, "archive.gz")
        self.assertEqual(validated.steps[0].command_template, "gzip <filename> archive.gz")

    def test_compression_with_source_but_no_destination_requests_destination(self):
        candidate = CandidateCommand(
            "filesystem:1", "filesystem", 1, "compress file", "gzip", "gzip <filename>", 0.8
        )
        plan = self.planner.plan(self.analyzer.analyze("compress /tmp/archive.tar"))
        proposal = PipelineBuilder(SelectionConfig(
            min_score=0.45,
            alternative_margin=0.04,
            max_alternatives=3,
        )).build(plan, {"step_1": [candidate]})

        validated = ProposalValidator().validate(proposal)

        self.assertEqual(validated.requires_input, ("destination",))

    def test_redirection_without_target_requests_destination(self):
        candidate = CandidateCommand(
            "filesystem:1", "filesystem", 1, "list files", "find", "find .", 0.8
        )
        plan = self.planner.plan(self.analyzer.analyze("find files >"))
        proposal = PipelineBuilder(SelectionConfig(
            min_score=0.45,
            alternative_margin=0.04,
            max_alternatives=3,
        )).build(plan, {"step_1": [candidate]})

        validated = ProposalValidator().validate(proposal)

        self.assertEqual(validated.requires_input, ("destination",))
        self.assertEqual(validated.steps[0].redirect_operator, ">")
        self.assertIsNone(validated.steps[0].redirect_target)

    def test_provided_source_satisfies_filename_but_not_destination(self):
        candidate = CandidateCommand(
            "filesystem:1", "filesystem", 1, "compress file", "gzip", "gzip <filename>", 0.8
        )
        plan = self.planner.plan(self.analyzer.analyze("compress /tmp/archive.tar"))
        proposal = PipelineBuilder(SelectionConfig(
            min_score=0.45,
            alternative_margin=0.04,
            max_alternatives=3,
        )).build(plan, {"step_1": [candidate]})

        validated = ProposalValidator().validate(proposal)

        self.assertEqual(validated.requires_input, ("destination",))
        self.assertEqual(validated.steps[0].command_template, "gzip /tmp/archive.tar")

    def test_prompt_builder_surfaces_missing_input(self):
        candidate = CandidateCommand(
            "filesystem:1", "filesystem", 1, "compress file", "gzip", "gzip <filename>", 0.8
        )
        plan = self.planner.plan(self.analyzer.analyze("compress file"))
        proposal = PipelineBuilder(SelectionConfig(
            min_score=0.45,
            alternative_margin=0.04,
            max_alternatives=3,
        )).build(plan, {"step_1": [candidate]})
        validated = ProposalValidator().validate(proposal)

        prompt = PromptBuilder().clarification(validated)

        self.assertIn("destination, filename", prompt)
        self.assertIn("Falta información para step_1", prompt)

    def test_ambiguity_is_reported_instead_of_filling_a_path(self):
        candidate = CandidateCommand(
            "filesystem:1", "filesystem", 1, "compare paths", "diff", "diff <path>", 0.8
        )
        plan = self.planner.plan(self.analyzer.analyze("compare /var/log with /tmp/logs"))
        proposal = PipelineBuilder(SelectionConfig(
            min_score=0.45,
            alternative_margin=0.04,
            max_alternatives=3,
        )).build(plan, {"step_1": [candidate]})

        validated = ProposalValidator().validate(proposal)

        self.assertEqual(validated.requires_input, ("path",))
        self.assertEqual(proposal.steps[0].command_template, "diff <path>")

    def test_spanish_multistep_with_arguments_and_dependencies(self):
        plan = self.planner.plan(self.analyzer.analyze(
            "buscar archivos en /var/log mayores a +100M y luego comprimir en backup.tar.gz"
        ))

        self.assertEqual(len(plan.steps), 2)
        self.assertEqual(plan.steps[0].arguments, (("path", "/var/log"), ("size", "+100M")))
        self.assertEqual(plan.steps[1].arguments, (("filename", "backup.tar.gz"),))
        self.assertEqual(plan.steps[1].depends_on, ("step_1",))
        self.assertEqual(plan.steps[1].connector_before, "natural")


if __name__ == "__main__":
    unittest.main()
