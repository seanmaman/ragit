import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')
log = logging.getLogger(__name__)

import os
import json
import time

from llama_cloud import FilterOperator, MetadataFilter, MetadataFilters

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from modules.ntt_utils import NTTUtils
from modules.ntt_rag import NTTRAG


def test():
    #prompt = "who and why froze weapons shipments to israel"
    prompt = "get me the link to get data for conifuguration in leaderboard"
    log.debug("quering...")

    ##response = nttrag.query(prompt)
    metadata_filters = MetadataFilters( filters=[  MetadataFilter(key="client",operator=FilterOperator.EQUAL_TO, value="lulu"),  ],)
    result = nttrag.query_with_sources(prompt)#,metadata_filters)
    nttrag.print_response_with_sources(prompt,result)

if __name__ == "__main__":
    
    with open("settings.json", "r") as f:
        settings = json.load(f)

    NTTUtils.print_info()

    #NTTUtils.init_whisper(settings["whisper_model"])

    nttrag = NTTRAG(settings["rag"])

    #nttrag.rebuild("documents")
    
    test()


    def is_valid_file(full_path):
        if not os.path.isfile(full_path):
            return False
        filename = os.path.basename(full_path)
        if filename.startswith('.') or filename.endswith('~') or filename == 'Thumbs.db':
            # Skip temporary/hidden files
            return False
        return True
    
    class Watcher(FileSystemEventHandler):
        def on_created(self, event):
            full_path = os.path.abspath(event.src_path)
            if not is_valid_file(full_path):
                return 
            if NTTUtils.wait_for_file_ready(full_path):
                log.info(f"file watcher - created: {full_path}")
                nttrag.add_file(full_path)

        def on_modified(self, event):
            full_path = os.path.abspath(event.src_path)
            if not is_valid_file(full_path):
                return 
            if NTTUtils.wait_for_file_ready(full_path):
                log.info(f"file watcher - modified: {full_path}")
                nttrag.add_file(full_path)
        
        def on_deleted(self, event):
            full_path = os.path.abspath(event.src_path)
            if not is_valid_file(full_path):
                return 
            log.info(f"file watcher - deleted: {full_path}")
            nttrag.delete_documents(full_path)

    event_handler = Watcher()
    observer = Observer()
    observer.schedule(event_handler, settings["documents_dir"], recursive=True)
    observer.start()
    log.info(f'monitoring documents direcory {settings["documents_dir"]}...')
        
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()