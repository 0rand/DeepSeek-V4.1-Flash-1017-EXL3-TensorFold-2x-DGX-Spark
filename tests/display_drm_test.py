#!/usr/bin/env python3
import os
import pathlib
import subprocess
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "display_drm.sh"


class DisplayDrmTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        base = pathlib.Path(self.tmp.name)
        self.sysfs = base / "sys"
        self.proc = base / "proc"
        self.bin = base / "bin"
        self.log = base / "commands.log"
        self.sysfs.mkdir()
        self.proc.mkdir()
        self.bin.mkdir()
        self.make_card()
        self.stub("id", '#!/bin/sh\nprintf "%s\\n" "${TEST_UID:-0}"\n')
        self.stub("nvidia-smi", '#!/bin/sh\n[ "${SMI_FAIL:-0}" = 1 ] && exit 1\nprintf "%s" "${COMPUTE_PIDS:-}"\n')
        self.stub("rmmod", '#!/bin/sh\nprintf "rmmod %s\\n" "$*" >> "$COMMAND_LOG"\n')
        self.stub("modprobe", '#!/bin/sh\nprintf "modprobe %s\\n" "$*" >> "$COMMAND_LOG"\nif [ "${MODPROBE_FAIL:-}" = "$*" ]; then exit 1; fi\nexit 0\n')
        self.stub("sleep", '#!/bin/sh\n: \n')

    def tearDown(self):
        self.tmp.cleanup()

    def stub(self, name, body):
        path = self.bin / name
        path.write_text(body)
        path.chmod(0o755)

    def make_card(self, connector=True, enabled="disabled"):
        card = self.sysfs / "class/drm/card7"
        device = card / "device"
        device.mkdir(parents=True, exist_ok=True)
        (device / "vendor").write_text("0x10de\n")
        if connector:
            output = self.sysfs / "class/drm/card7-DP-1"
            output.mkdir(parents=True, exist_ok=True)
            (output / "status").write_text("connected\n")
            if enabled is not None:
                (output / "enabled").write_text(enabled + "\n")

    def make_render_node(self, number=128):
        device = self.sysfs / f"class/drm/renderD{number}/device"
        device.mkdir(parents=True, exist_ok=True)
        (device / "vendor").write_text("0x10de\n")

    def invoke(self, action, **env):
        run_env = os.environ | {
            "PATH": f"{self.bin}:{os.environ['PATH']}",
            "COMMAND_LOG": str(self.log),
            "TF_DISPLAY_DRM_SYSFS": str(self.sysfs),
            "TF_DISPLAY_DRM_PROC_ROOT": str(self.proc),
            "TF_DISPLAY_DRM_MODULES": str(self.sysfs / "module"),
        } | env
        return subprocess.run(["bash", str(SCRIPT), action], text=True,
                              capture_output=True, env=run_env, check=False)

    def commands(self):
        return self.log.read_text() if self.log.exists() else ""

    def map_process(self, pid, content):
        maps = self.proc / str(pid) / "maps"
        maps.parent.mkdir()
        maps.write_text(content + "\n")

    def open_fd(self, pid, number, target):
        fd = self.proc / str(pid) / "fd" / str(number)
        fd.parent.mkdir(parents=True, exist_ok=True)
        fd.symlink_to(target)

    def test_compute_process_refuses_without_commands(self):
        result = self.invoke("on", COMPUTE_PIDS="123\n")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("compute process", result.stderr)
        self.assertEqual(self.commands(), "")

    def test_live_drm_map_refuses_without_commands(self):
        self.map_process(321, "7f00 /dev/dri/card7")
        result = self.invoke("off")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("pid 321 maps an NVIDIA DRM device", result.stderr)
        self.assertEqual(self.commands(), "")

    def test_display_kv_map_refuses_without_commands(self):
        self.map_process(322, "7f00 /srv/libdisplay_kv.so")
        result = self.invoke("on")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("libdisplay_kv", result.stderr)
        self.assertEqual(self.commands(), "")

    def test_open_card_fd_without_mapping_refuses(self):
        self.open_fd(323, 9, "/dev/dri/card7")
        result = self.invoke("off")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("pid 323 has an NVIDIA DRM device open", result.stderr)
        self.assertEqual(self.commands(), "")

    def test_render_node_maps_and_fds_refuse(self):
        self.make_render_node()
        self.map_process(324, "7f00 /dev/dri/renderD128")
        mapped = self.invoke("on")
        self.assertNotEqual(mapped.returncode, 0)
        self.assertIn("maps an NVIDIA DRM device", mapped.stderr)
        (self.proc / "324").rename(self.proc / "325")
        (self.proc / "325/maps").unlink()
        self.open_fd(325, 10, "/dev/dri/renderD128")
        opened = self.invoke("on")
        self.assertNotEqual(opened.returncode, 0)
        self.assertIn("pid 325 has an NVIDIA DRM device open", opened.stderr)
        self.assertEqual(self.commands(), "")

    def test_positive_module_refcount_refuses(self):
        refcount = self.sysfs / "module/nvidia_drm/refcnt"
        refcount.parent.mkdir(parents=True)
        refcount.write_text("2\n")
        result = self.invoke("on")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("module refcount is 2", result.stderr)
        self.assertEqual(self.commands(), "")

    def test_unrelated_framebuffer_fd_is_allowed(self):
        device = self.sysfs / "class/drm/card0/device"
        device.mkdir(parents=True)
        (device / "vendor").write_text("0x1234\n")
        self.open_fd(326, 4, "/dev/dri/card0")
        result = self.invoke("off")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.commands(), "rmmod nvidia_drm\nmodprobe nvidia_drm\n")

    def test_active_and_unknown_connector_refuse(self):
        (self.sysfs / "class/drm/card7-DP-1/enabled").write_text("enabled\n")
        active = self.invoke("on")
        self.assertNotEqual(active.returncode, 0)
        self.assertIn("active NVIDIA scanout", active.stderr)
        (self.sysfs / "class/drm/card7-DP-1/enabled").unlink()
        unknown = self.invoke("on")
        self.assertNotEqual(unknown.returncode, 0)
        self.assertIn("state is unknown", unknown.stderr)
        self.assertEqual(self.commands(), "")

    def test_connected_disabled_is_allowed_and_uses_expected_arguments(self):
        on = self.invoke("on")
        self.assertEqual(on.returncode, 0, on.stderr)
        off = self.invoke("off")
        self.assertEqual(off.returncode, 0, off.stderr)
        self.assertEqual(self.commands(), "rmmod nvidia_drm\nmodprobe nvidia_drm modeset=1 fbdev=0\nrmmod nvidia_drm\nmodprobe nvidia_drm\n")

    def test_modprobe_failure_restores_default(self):
        result = self.invoke("on", MODPROBE_FAIL="nvidia_drm modeset=1 fbdev=0")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("failed to load", result.stderr)
        self.assertEqual(self.commands(), "rmmod nvidia_drm\nmodprobe nvidia_drm modeset=1 fbdev=0\nmodprobe nvidia_drm\n")

    def test_no_nvidia_card_refuses(self):
        (self.sysfs / "class/drm/card7/device/vendor").write_text("0x1234\n")
        result = self.invoke("on")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("no NVIDIA DRM card", result.stderr)
        self.assertEqual(self.commands(), "")

    def test_nvidia_driver_on_later_uevent_line_is_discovered(self):
        device = self.sysfs / "class/drm/card7/device"
        (device / "vendor").write_text("0x1234\n")
        (device / "uevent").write_text("PCI_CLASS=030000\nDRIVER=nvidia\n")
        result = self.invoke("off")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.commands(), "rmmod nvidia_drm\nmodprobe nvidia_drm\n")

    def test_root_is_required_for_changes_but_not_check(self):
        change = self.invoke("off", TEST_UID="1000")
        self.assertNotEqual(change.returncode, 0)
        self.assertIn("require root", change.stderr)
        self.assertEqual(self.commands(), "")
        check = self.invoke("check", TEST_UID="1000")
        self.assertEqual(check.returncode, 0, check.stderr)
        self.assertIn("card=card7", check.stdout)


if __name__ == "__main__":
    unittest.main()
