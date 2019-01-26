import scrapy
import json
import os
import re

out_dir = os.getcwd + '\\data'
base_url = 'https://sentence.yourdictionary.com/'

class SentenceSpider(scrapy.Spider):
    name = "sentbot"
    start_urls = [base_url]

    def parse(self, response):

        links = response.xpath('//*[@id="browse_section"]/div[@class="definitions_slider"]/ul[@class="bxslider"]/li/span/a/@href').extract()

        for link in links:
            sub_url = base_url + link
            yield response.follow(sub_url, callback=self.parse_sublinks, meta={"sub_dirname": link.split('/')[2]})


    def parse_sublinks(self, response):
        links = response.xpath('//*/div[@id="content_top"]/div[@class!="pagination-centered index_pagination"]/ul/li/a/@href').extract()
        
        save_dir = os.path.join(out_dir, response.meta.get("sub_dirname"))

        if not os.path.exists(save_dir):
            os.mkdir(save_dir)


        for link in links:
            word = link.replace('/','')
            absolute_url = base_url + word
            file_path = os.path.join(save_dir, word)
            yield response.follow(absolute_url, callback=self.parse_abs_links, meta={"file_path": file_path})


    def parse_abs_links(self, response):
        sents = response.xpath('//*[@id="examples-ul-content"]/li/div[@class="li_content"]').extract()

        with open(response.meta.get('file_path'), 'w') as f:
            for s  in sents:
                f.write(re.sub('<[^>]+>', '', s) + '\n')
        
        yield None
