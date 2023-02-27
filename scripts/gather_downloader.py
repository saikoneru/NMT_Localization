import argparse
import asyncio
import logging
import sys
import codecs

from aiopath import AsyncPath
from utils_client import create_client, setup_logging


class GitHubDownloader:
    def __init__(self, client, output_path):
        self._client = client
        self.output_path = output_path

    @staticmethod
    def _url_to_path(url):
        if 'https://raw.githubusercontent.com/' in url:
            return url.replace("https://raw.githubusercontent.com/", "")
        if 'https://gitlab.com/' in url:
            return url.replace("https://gitlab.com/", "")

    async def download(self, url):
        async with self._client.get(url) as resp:
            return await resp.text()

    async def save(self, url, content):
        path = self.output_path / AsyncPath(self._url_to_path(url))
        await path.parent.mkdir(parents=True, exist_ok=True)
        await path.write_text(content)

    async def download_and_save(self, url):
        logging.info("Downloading %s", url)
        try:
            content = await self.download(url)
        except Exception as e:
            print(url)
            logging.exception("Exception occurred while downloading: %s", str(e))
            await asyncio.sleep(0.5)
            return
        #logging.info("Saving %s", url)
        try:
            await self.save(url, content)
        except Exception as e:
            logging.exception("Exception occurred while saving: %s", str(e))

    async def fetch(self,url):
        async with self._client.get(url) as resp:
            if resp.status == 429:
                print(url)
                sys.exit(1)
            try:
                content = await resp.text()
            except:
                content = await resp.text(encoding = 'iso-8859-1')
            await self.save(url,content)

    async def fetch_all(self,urls):
        tasks = []
        for url in urls:
            task = asyncio.create_task(self.fetch(url))
            tasks.append(task)
        results = await asyncio.gather(*tasks)
        return results

async def main():
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o", "--output", help="Output file path (default: 'out')", default="out"
    )
    parser.add_argument(
        "-l",
        dest="limit",
        help="Concurrent requests limit (default: 1000)",
        default=1000,
        type=int,
    )
    args = parser.parse_args()
    async with create_client(request_limit=args.limit) as client:
        downloader = GitHubDownloader(client=client, output_path=args.output)
        tasks = []
        urls = (l.rstrip("\n") for l in sys.stdin)
        res = await downloader.fetch_all(urls)

if __name__ == "__main__":
    asyncio.run(main())
