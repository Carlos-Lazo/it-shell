import unittest
from semantic_search_engine.shared.contracts import (
    CandidateCommand,
    PipelineProposal,
    PipelineStep,
)
from semantic_search_engine.pipeline_builder_service.validator import ProposalValidator


class TestSafetyValidator(unittest.TestCase):
    def setUp(self):
        self.validator = ProposalValidator()

    def _make_proposal(self, template: str, command: str = "test") -> PipelineProposal:
        candidate = CandidateCommand("cat:1", "cat", 1, "intent", command, template, 0.8)
        step = PipelineStep("step_1", "query", (), candidate, (), template)
        return PipelineProposal("goal", (step,), executable=False)

    def test_safe_commands_matrix(self):
        safe_commands = [
            "ls -la",
            "find /var/log -name '*.log'",
            "grep -i 'error' /var/log/syslog",
            "cat /etc/hosts",
            "df -h",
            "du -sh /home/*",
            "ps aux",
            "head -n 20 access.log",
            "tail -f /var/log/messages",
            "tar -czf archive.tar.gz file1 file2",
            "curl -O https://example.com/file.txt",
        ]
        for cmd in safe_commands:
            with self.subTest(cmd=cmd):
                prop = self.validator.validate(self._make_proposal(cmd))
                self.assertFalse(prop.executable)
                self.assertEqual(len(prop.issues), 0, f"Expected 0 issues for safe cmd '{cmd}', got {prop.issues}")

    def test_destructive_commands_matrix(self):
        destructive_commands = [
            ("rm -rf /tmp/data", "destructive_command"),
            ("rm -fr /tmp/data", "destructive_command"),
            ("shred -u secret.txt", "destructive_command"),
            ("wipefs -a /dev/sdb", "destructive_command"),
            ("mkfs.ext4 /dev/sdb1", "destructive_command"),
            ("fdisk /dev/sda", "destructive_command"),
            ("dd if=/dev/zero of=/dev/sdb bs=1M", "destructive_command"),
            ("cat image.iso > /dev/sdb", "destructive_command"),
        ]
        for cmd, expected_code in destructive_commands:
            with self.subTest(cmd=cmd):
                prop = self.validator.validate(self._make_proposal(cmd))
                self.assertFalse(prop.executable)
                codes = [i.code for i in prop.issues]
                self.assertIn(expected_code, codes, f"Expected {expected_code} for '{cmd}', got {codes}")

    def test_privileged_commands_matrix(self):
        privileged_commands = [
            ("sudo systemctl restart nginx", "privileged_command"),
            ("su - root", "privileged_command"),
            ("shutdown -h now", "privileged_command"),
            ("reboot", "privileged_command"),
            ("poweroff", "privileged_command"),
            ("chmod -R 755 /var/www", "privileged_command"),
            ("chmod 777 script.sh", "privileged_command"),
            ("chown -R www-data:www-data /var/www", "privileged_command"),
            ("iptables -F", "privileged_command"),
            ("ufw disable", "privileged_command"),
            ("kill -9 1234", "privileged_command"),
        ]
        for cmd, expected_code in privileged_commands:
            with self.subTest(cmd=cmd):
                prop = self.validator.validate(self._make_proposal(cmd))
                self.assertFalse(prop.executable)
                codes = [i.code for i in prop.issues]
                self.assertIn(expected_code, codes, f"Expected {expected_code} for '{cmd}', got {codes}")

    def test_combined_destructive_and_privileged(self):
        cmd = "sudo rm -rf /var/cache/*"
        prop = self.validator.validate(self._make_proposal(cmd))
        self.assertFalse(prop.executable)
        codes = [i.code for i in prop.issues]
        self.assertIn("destructive_command", codes)
        self.assertIn("privileged_command", codes)

    def test_placeholder_extraction(self):
        cmd = "find <directory> -name <filename>"
        prop = self.validator.validate(self._make_proposal(cmd))
        self.assertEqual(prop.requires_input, ("directory", "filename"))
        self.assertFalse(prop.executable)

    def test_guarantee_never_executable(self):
        # Even with executable=True initially passed in proposal, validator sets False
        proposal = PipelineProposal("goal", (), executable=True)
        validated = self.validator.validate(proposal)
        self.assertFalse(validated.executable)


if __name__ == "__main__":
    unittest.main()
