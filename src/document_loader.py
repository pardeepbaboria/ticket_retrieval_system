import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Any
from langchain_core.documents import Document
from langchain_community.document_loaders import JSONLoader
from functools import partial

logger = logging.getLogger(__name__)

class SupportDocumentLoader:
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.seen_ticket_ids = set()

    def get_json_content(self, data: Dict[str, Any]) -> str:
        return (
            f"Subject: {data.get('subject', '')}\n"
            f"Description: {data.get('body', '')}\n"
            f"Resolution: {data.get('answer', '')}\n"
            f"Type: {data.get('type', '')}\n"
            f"Queue: {data.get('queue', '')}\n"
            f"Priority: {data.get('priority', '')}"
        )

    def get_json_metadata(self, record: Dict[str, Any], support_type: str = None) -> Dict[str, Any]:
        if not support_type:
            raise ValueError("support_type is not provided")

        original_id = str(record.get('Ticket ID', ''))
        ticket_id = f"{support_type}_{original_id}"

        if ticket_id in self.seen_ticket_ids:
            raise ValueError(f"Duplicate ticket ID found: {ticket_id}")
        self.seen_ticket_ids.add(ticket_id)

        tags = []
        for i in range(1, 9):
            tag_val = record.get(f'tag_{i}')
            if tag_val and str(tag_val).lower() != 'nan':
                tags.append(str(tag_val))

        return {
            'ticket_id': ticket_id,
            'original_ticket_id': original_id,
            'support_type': support_type,
            'type': str(record.get('type', '')),
            'queue': str(record.get('queue', '')),
            'priority': str(record.get('priority', '')),
            'language': str(record.get('language', '')),
            'tags': tags,
            'source': 'json',
            'subject': str(record.get('subject', '')),
            'body': str(record.get('body', '')),
            'answer': str(record.get('answer', ''))
        }

    def _metadata_transform(self, record, metadata=None, support_type=None):
        return self.get_json_metadata(record, support_type=support_type)

    def load_xml_tickets(self, file_path: Path, support_type: str) -> List[Document]:
        documents = []
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()

            for ticket_elem in root.findall('.//Ticket'):
                original_id = ticket_elem.findtext('TicketID', '')
                ticket_id = f"{support_type}_xml_{original_id}"

                if ticket_id in self.seen_ticket_ids:
                    raise ValueError(f"Duplicate ticket ID found: {ticket_id}")
                self.seen_ticket_ids.add(ticket_id)

                content = (
                    f"Subject: {ticket_elem.findtext('subject', '')}\n"
                    f"Description: {ticket_elem.findtext('body', '')}\n"
                    f"Resolution: {ticket_elem.findtext('answer', '')}\n"
                    f"Type: {ticket_elem.findtext('type', '')}\n"
                    f"Queue: {ticket_elem.findtext('queue', '')}\n"
                    f"Priority: {ticket_elem.findtext('priority', '')}"
                )

                tags = []
                for i in range(1, 9):
                    tag_val = ticket_elem.findtext(f'tag_{i}')
                    if tag_val and str(tag_val).lower() != 'nan':
                        tags.append(str(tag_val))

                metadata = {
                    'ticket_id': ticket_id,
                    'original_ticket_id': original_id,
                    'support_type': support_type,
                    'type': ticket_elem.findtext('type', ''),
                    'queue': ticket_elem.findtext('queue', ''),
                    'priority': ticket_elem.findtext('priority', ''),
                    'language': ticket_elem.findtext('language', ''),
                    'tags': tags,
                    'source': 'xml'
                }

                documents.append(Document(page_content=content, metadata=metadata))
        except Exception as e:
            logger.error(f"Error parsing XML file {file_path}: {e}")
        
        return documents

    def load_tickets(self) -> Dict[str, List[Document]]:
        all_docs = {"technical": [], "product": [], "customer": []}
        
        mapping = {
            "Technical Support_tickets": "technical",
            "Product Support_tickets": "product",
            "Customer Service_tickets": "customer"
        }

        for file in self.data_path.iterdir():
            stem = file.stem
            if stem in mapping:
                s_type = mapping[stem]
                
                if file.suffix == '.json':
                    
                    loader = JSONLoader(
                        file_path=str(file),
                        jq_schema='.[]',
                        content_key=None,
                        text_content=False,
                        metadata_func=partial(self._metadata_transform, support_type=s_type)
                    )

                    docs = loader.load()

                    for d in docs:
                        d.page_content = self.get_json_content(d.metadata)
                        all_docs[s_type].append(d)

                elif file.suffix == '.xml':
                    all_docs[s_type].extend(self.load_xml_tickets(file, s_type))

        return all_docs

    def create_documents(self) -> Dict[str, List[Document]]:
        return self.load_tickets()