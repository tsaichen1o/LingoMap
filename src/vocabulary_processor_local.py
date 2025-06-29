#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# python src/vocabulary_processor.py
"""
LingoMap Vocabulary Processor (v3.1 - Local DB Only with Deduplication)
Reads vocabulary from local Turtle files and stores them in a local ChromaDB instance.
This version is designed to bypass cloud quota limitations for development and handles duplicate IDs gracefully.
"""
import chromadb
from rdflib import Graph, RDFS, SKOS, URIRef, Namespace
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import os
from dotenv import load_dotenv

load_dotenv()

class VocabularyProcessor:
    """
    Processes RDF vocabulary files into a searchable vector database.
    """

    def __init__(self, model_name: str = 'all-mpnet-base-v2'):
        print(f"🚀 Initializing Vocabulary Processor...")
        # 1. 初始化嵌入模型 (只在需要時下載一次)
        print(f"   - Loading embedding model: {model_name}...")
        self.embedding_model = SentenceTransformer(model_name)
        
        # --- [MODIFIED] 強制使用本地端儲存 ---
        print("   - Initializing ChromaDB client in Local Mode...")
        self._init_local_client()
        # ------------------------------------

        collection_name = "lingomap_vocab"
        print(f"   - Getting or creating collection: {collection_name}")
        self.collection = self.client.get_or_create_collection(name=collection_name)

        print("   - Fetching existing IDs from the local collection...")
        try:
            # 使用 get() 方法，只請求 id，避免載入大量數據
            existing_items = self.collection.get(include=[]) 
            self.processed_ids = set(existing_items['ids'])
            print(f"   - Found {len(self.processed_ids)} existing IDs in the local database.")
        except Exception as e:
            print(f"   - Warning: Could not fetch existing IDs. May re-process some items. Error: {e}")
            self.processed_ids = set()
            
        print("✅ Initialization complete.")
        
    def _init_local_client(self):
        """一個輔助函式，用於初始化本機端的 ChromaDB Client。"""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        db_path = os.path.join(project_root, "lingomap_db")
        print(f"   - Database will be stored at: {db_path}")
        self.client = chromadb.PersistentClient(path=db_path)

    def process_turtle_file(self, file_path: str):
        """
        Reads a Turtle file, extracts terms and a RICH set of descriptions, 
        and adds them to the database.
        """
        if not os.path.exists(file_path):
            print(f"❌ Error: File not found at {file_path}")
            return

        print(f"\n📄 Processing file: {file_path}...")
        g = Graph()
        # 簡化解析邏輯，rdflib 會自動猜測格式
        try:
            g.parse(file_path)
        except Exception as e:
            print(f"   - ❌ Error parsing file {file_path}: {str(e)}")
            return

        documents_to_add, metadatas_to_add, ids_to_add = [], [], []

        def get_label(subject, g):
            """Helper to get the label of a subject in the graph."""
            label = g.value(subject, RDFS.label) or g.value(subject, SKOS.prefLabel)
            return str(label) if label else None

        for subject in g.subjects():
            if isinstance(subject, URIRef):
                subject_str = str(subject)
                # 跳過已在資料庫中或先前檔案中處理過的ID
                if subject_str in self.processed_ids:
                    continue

                label = get_label(subject, g)
                definition = g.value(subject, SKOS.definition) or g.value(subject, RDFS.comment)

                if label and definition:
                    # --- 建立豐富上下文的邏輯 (維持不變) ---
                    context_parts = []
                    for super_prop in g.objects(subject, RDFS.subPropertyOf):
                        if super_label := get_label(super_prop, g):
                            context_parts.append(f"Is a type of: {super_label}")
                    for super_class in g.objects(subject, RDFS.subClassOf):
                        if super_label := get_label(super_class, g):
                             context_parts.append(f"Is a subclass of: {super_label}")
                    for range_class in g.objects(subject, RDFS.range):
                        if range_label := get_label(range_class, g):
                            context_parts.append(f"Expected data type: {range_label}")
                    CMNS_AV = Namespace("https://www.omg.org/spec/Commons/AnnotationVocabulary/")
                    for note in g.objects(subject, CMNS_AV.usageNote):
                        context_parts.append(f"Usage note: {str(note)}")
                    
                    document_text = f"Term: {label}\nDefinition: {str(definition)}"
                    if context_parts:
                        document_text += "\n\nContext:\n- " + "\n- ".join(context_parts)
                    # ------------------------------------------

                    documents_to_add.append(document_text)
                    metadatas_to_add.append({
                        "uri": subject_str, "label": str(label), "source_file": os.path.basename(file_path)
                    })
                    ids_to_add.append(subject_str)

        if not documents_to_add:
            print("   - No new suitable terms found in this file.")
            return
        
        # --- [FIX] 在添加到資料庫之前，對當前檔案的結果進行去重 ---
        final_docs, final_metas, final_ids = [], [], []
        seen_ids_in_file = set()

        for i in range(len(ids_to_add)):
            doc_id = ids_to_add[i]
            if doc_id not in seen_ids_in_file:
                seen_ids_in_file.add(doc_id)
                final_docs.append(documents_to_add[i])
                final_metas.append(metadatas_to_add[i])
                final_ids.append(doc_id)
        # --- [END FIX] ---

        if not final_ids:
            print("   - All found terms in this file were duplicates, nothing new to add.")
            return

        # 使用去重後的列表進行資料庫操作
        print(f"   - Found {len(final_ids)} unique new terms. Adding to the local database...")
        try:
            self.collection.add(
                documents=final_docs,
                metadatas=final_metas,
                ids=final_ids
            )
            # 更新已處理 ID 集合
            self.processed_ids.update(final_ids)
            print(f"✅ Successfully added {len(final_ids)} terms from {file_path}.")
        except Exception as e:
            print(f"   - ❌ An error occurred while adding to the local database: {e}")

    def search(self, query_text: str, n_results: int = 5) -> List[Dict]:
        """
        Searches the vector database for the most similar terms.
        """
        print(f"\n🔍 Searching for terms similar to: '{query_text}'...")
        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results
        )

        cleaned_results = []
        if results and results.get('metadatas') and results['metadatas'][0]:
            for i, metadata in enumerate(results['metadatas'][0]):
                cleaned_results.append({
                    "label": metadata.get("label"),
                    "uri": metadata.get("uri"),
                    "distance": results['distances'][0][i]
                })

        return cleaned_results


def main():
    """主函數，用於展示整個流程"""
    try:
        processor = VocabularyProcessor()
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        vocab_directory = os.path.join(project_root, "vocabularies")

        if not os.path.exists(vocab_directory):
            print(f"❌ 錯誤：找不到 '{vocab_directory}' 資料夾。")
            return

        ttl_files_to_process = [
            os.path.join(root, file) 
            for root, _, files in os.walk(vocab_directory) 
            for file in files if file.endswith((".ttl", ".rdf", ".owl"))
        ]

        if not ttl_files_to_process:
            print(f"⚠️ 警告：在 '{vocab_directory}' 資料夾中沒有找到任何詞彙庫檔案。")
        else:
            print(f"📂 發現 {len(ttl_files_to_process)} 個詞彙庫檔案，準備處理...")
            for ttl_file in sorted(ttl_files_to_process):
                processor.process_turtle_file(ttl_file)

        # 測試搜尋
        column_name = "CERT"
        sample_data = [14761.0, 57899.0]
        query = f"Column Name: {column_name}\nDescription: A certificate number for a financial institution.\nSample Values: {sample_data}"

        search_results = processor.search(query, n_results=5)

        print("\n📊 Test Search Results:")
        if search_results:
            for result in search_results:
                print(f"  - Label: {result['label']}\n    URI: {result['uri']}\n    Distance: {result['distance']:.4f}\n")
        else:
            print("   - No results found.")
    except Exception as e:
        print(f"An error occurred in main: {e}")


if __name__ == "__main__":
    main()
