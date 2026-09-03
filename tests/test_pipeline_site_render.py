import unittest

from etf_radar import pipeline


class PipelineSiteReRenderTests(unittest.TestCase):
    def test_rerender_called_when_callback_present(self):
        calls = []
        previous = pipeline._core._LAST_SITE_RENDER_CALLBACK
        try:
            pipeline._core._LAST_SITE_RENDER_CALLBACK = lambda: calls.append(True)
            pipeline._re_render_site_after_distribution_audit()
            self.assertEqual(calls, [True])
        finally:
            pipeline._core._LAST_SITE_RENDER_CALLBACK = previous

    def test_missing_callback_is_a_noop(self):
        previous = pipeline._core._LAST_SITE_RENDER_CALLBACK
        try:
            pipeline._core._LAST_SITE_RENDER_CALLBACK = None
            pipeline._re_render_site_after_distribution_audit()  # must not raise
        finally:
            pipeline._core._LAST_SITE_RENDER_CALLBACK = previous

    def test_failed_rerender_raises(self):
        previous = pipeline._core._LAST_SITE_RENDER_CALLBACK
        try:
            def broken_render():
                raise RuntimeError("boom")

            pipeline._core._LAST_SITE_RENDER_CALLBACK = broken_render
            with self.assertRaises(RuntimeError):
                pipeline._re_render_site_after_distribution_audit()
        finally:
            pipeline._core._LAST_SITE_RENDER_CALLBACK = previous


if __name__ == "__main__":
    unittest.main()
