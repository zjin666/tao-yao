import asyncio
import unittest
from unittest import mock

from taoyao.linkreader import BILI_ID_RE, LinkReader, LinkResult


class BilibiliLinkTests(unittest.TestCase):
    def test_ids_and_hosts_are_recognised(self):
        self.assertEqual(BILI_ID_RE.search("/video/BV17x411w7KC/").group(1), "BV17x411w7KC")
        self.assertEqual(BILI_ID_RE.search("/video/av170001").group(1), "av170001")
        self.assertTrue(LinkReader._is_bilibili("https://b23.tv/abc"))
        self.assertTrue(LinkReader._is_bilibili("https://www.bilibili.com/video/BV17x411w7KC"))
        self.assertFalse(LinkReader._is_bilibili("https://evilbilibili.com/video/BV17x411w7KC"))

    def test_bilibili_links_use_the_site_reader_and_fall_back_when_it_gives_up(self):
        reader = LinkReader({})
        got = LinkResult("u", title="标题", text="B站视频")
        with mock.patch.object(reader, "_safe_host", return_value=(True, "")), \
                mock.patch.object(reader, "_fetch_bilibili", mock.AsyncMock(return_value=got)):
            self.assertIs(asyncio.run(reader.fetch("https://b23.tv/x")), got)


class VideoWatchTests(unittest.IsolatedAsyncioTestCase):
    def test_query_strings_are_kept_in_links(self):
        from taoyao.linkreader import extract_links
        self.assertEqual(extract_links("看 https://www.youtube.com/watch?v=abc&t=5？好看吗"),
                         ["https://www.youtube.com/watch?v=abc&t=5"])

    def test_lowest_quality_video_stream_is_picked_and_audio_only_skipped(self):
        from taoyao.videolink import pick_stream, frame_times
        formats = [{"url": "a", "vcodec": "none", "protocol": "https"},
                   {"url": "hd", "vcodec": "avc1", "protocol": "https", "height": 1080},
                   {"url": "sd", "vcodec": "avc1", "protocol": "m3u8_native", "height": 360},
                   {"url": "dash", "vcodec": "avc1", "protocol": "http_dash_segments", "height": 144}]
        self.assertEqual(pick_stream(formats)["url"], "sd")
        self.assertEqual(frame_times(100, 4), [12.5, 37.5, 62.5, 87.5])

    async def test_watcher_reports_frames_and_is_off_without_ffmpeg(self):
        from pathlib import Path
        import tempfile
        from taoyao.videolink import VideoInfo, VideoWatcher
        info = VideoInfo("u", uploader="作者", description="简介", duration=100, stream_url="s")

        async def describe(frames, prompt):
            return "画面里有人"
        with tempfile.TemporaryDirectory() as tmp:
            w = VideoWatcher({}, Path("ffmpeg"), describe, Path(tmp))
            with mock.patch("taoyao.videolink.is_video_site", return_value=True),                     mock.patch("taoyao.videolink.resolve", return_value=info),                     mock.patch("taoyao.videolink.grab_frame", side_effect=lambda *a: a[3].write_bytes(b"x") or True):
                text = await w.watch("https://v/x")
                self.assertIn("画面里有人", text)
                self.assertIn("作者", text)
                self.assertNotIn("作者", await w.watch("https://v/x", with_meta=False))
            self.assertEqual(await VideoWatcher({}, None, describe, Path(tmp)).watch("https://v/x"), "")
            with mock.patch("taoyao.videolink.is_video_site", return_value=False):
                self.assertEqual(await w.watch("https://plain.example/page"), "")


if __name__ == "__main__":
    unittest.main()
