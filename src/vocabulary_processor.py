#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# python src/vocabulary_processor.py
"""
LingoMap Vocabulary Processor (v2 - Cloud Ready)
Reads vocabulary from Turtle files, generates embeddings, and stores them in ChromaDB.
Prioritizes ChromaDB Cloud connection and falls back to local persistence if not configured.
"""
import sys
import os

# Fix SQLite version issue for Streamlit Cloud deployment
try:
    import pysqlite3
    sys.modules['sqlite3'] = pysqlite3
except ImportError:
    pass

import chromadb
from rdflib import Graph, RDFS, SKOS, URIRef, Namespace
from sentence_transformers import SentenceTransformer
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()

class VocabularyProcessor:
    """
    Processes RDF vocabulary files into a searchable vector database.
    """

    def __init__(self, model_name: str = 'all-mpnet-base-v2', quota_limit: int = 128):
        print(f"🚀 Initializing Vocabulary Processor...")
        # 1. 初始化嵌入模型 (只在需要時下載一次)
        print(f"   - Loading embedding model: {model_name}...")
        self.embedding_model = SentenceTransformer(model_name)
        
        # 設定配額限制
        self.quota_limit = quota_limit
        print(f"   - ChromaDB quota limit set to: {quota_limit} bytes")
        
        # --- [MODIFIED] 初始化 ChromaDB，優先使用雲端版本 ---
        print("   - Initializing ChromaDB client...")
        
        # 從環境變數中讀取雲端設定
        chroma_api_key = os.getenv("CHROMADB_API_KEY")
        chroma_tenant = os.getenv("CHROMADB_TENANT_ID")
        chroma_database = os.getenv("CHROMADB_NAME")
        
        # 檢查是否所有必要的雲端變數都已設定
        if chroma_api_key and chroma_tenant and chroma_database:
            print("   - 發現雲端設定，正在嘗試連接到 ChromaDB Cloud...")
            try:
                self.client = chromadb.CloudClient(
                    tenant=chroma_tenant,
                    database=chroma_database,
                    api_key=chroma_api_key
                )
                print("   ✅ 成功連接到 ChromaDB Cloud!")
            except Exception as e:
                print(f"   ❌ 連接到 ChromaDB Cloud 失敗: {e}")
                print("   - 將退回使用本機端儲存。")
                self._init_local_client()
        else:
            print("   - 未找到完整的 ChromaDB Cloud 設定，將使用本機端儲存。")
            self._init_local_client()

        # 3. 創建或獲取一個 Collection (此部分邏輯不變)
        collection_name = "lingomap_vocab"
        print(f"   - Getting or creating collection: {collection_name}")
        self.collection = self.client.get_or_create_collection(name=collection_name)

        # 4. 跟踪已处理的ID，避免重复
        print("   - Fetching existing IDs from the collection to prevent duplicates...")
        try:
            # 使用 get() 方法，只請求 id，這樣可以處理大量的現有項目
            existing_items = self.collection.get(include=[]) 
            self.processed_ids = set(existing_items['ids'])
            print(f"   - Found {len(self.processed_ids)} existing IDs in the collection.")
        except Exception as e:
            print(f"   - Warning: Could not fetch existing IDs. May re-process some items. Error: {e}")
            self.processed_ids = set()
        print("✅ Initialization complete.")
        
    def _init_local_client(self):
        """一個輔助函式，用於初始化本機端的 ChromaDB Client。"""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        db_path = os.path.join(project_root, "lingomap_db")
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
        # ... (解析檔案的邏輯維持不變) ...
        file_ext = os.path.splitext(file_path)[1].lower()
        parse_format = "turtle" if file_ext == '.ttl' else "xml"
        try:
            g.parse(file_path, format=parse_format)
        except Exception as e:
            print(f"   - ❌ Error parsing file {file_path}: {str(e)}")
            return

        documents_to_add, metadatas_to_add, ids_to_add = [], [], []
        skipped_count = 0

        # --- 新的輔助函式，用來獲取關聯資源的標籤 ---
        def get_label(subject, g):
            """Helper to get the label of a subject in the graph."""
            label = g.value(subject, RDFS.label) or g.value(subject, SKOS.prefLabel)
            return str(label) if label else None

        for subject in g.subjects():
            if isinstance(subject, URIRef):
                subject_str = str(subject)
                if subject_str in self.processed_ids:
                    skipped_count += 1
                    continue

                label = get_label(subject, g)
                definition = g.value(subject, SKOS.definition) or g.value(subject, RDFS.comment)

                if label and definition:
                    # --- 核心修改：建立一個豐富的上下文列表 ---
                    context_parts = []
                    
                    # 1. 獲取父屬性/父類別的名稱
                    for super_prop in g.objects(subject, RDFS.subPropertyOf):
                        super_label = get_label(super_prop, g)
                        if super_label:
                            context_parts.append(f"Is a type of: {super_label}")

                    for super_class in g.objects(subject, RDFS.subClassOf):
                        super_label = get_label(super_class, g)
                        if super_label:
                             context_parts.append(f"Is a subclass of: {super_label}")
                    
                    # 2. 獲取值域(range)的名稱
                    for range_class in g.objects(subject, RDFS.range):
                        range_label = get_label(range_class, g)
                        if range_label:
                            context_parts.append(f"Expected data type: {range_label}")

                    # 3. 獲取其他筆記
                    # (注意：您需要綁定 cmns-av namespace 才能用 cmns-av:usageNote)
                    CMNS_AV = Namespace("https://www.omg.org/spec/Commons/AnnotationVocabulary/")
                    for note in g.objects(subject, CMNS_AV.usageNote):
                        context_parts.append(f"Usage note: {str(note)}")
                    
                    # --- 組合最終的、豐富的「語義文件」 ---
                    document_text = f"Term: {label}\nDefinition: {str(definition)}"
                    if context_parts:
                        document_text += "\n\nContext:\n- " + "\n- ".join(context_parts)
                    # ------------------------------------------

                    documents_to_add.append(document_text)
                    metadatas_to_add.append({
                        "uri": subject_str, "label": str(label), "source_file": os.path.basename(file_path)
                    })
                    ids_to_add.append(subject_str)
                    self.processed_ids.add(subject_str)

        # ... (後續加入 ChromaDB 的邏輯維持不變) ...
        if not documents_to_add:
            print("   - No suitable terms found.")
            return
        
        # --- [NEW] Smart Batching Logic with Dynamic Quota Management ---
        MAX_BATCH_ITEMS = 100  # 批次最大項目數
        MAX_BATCH_BYTES = 100 * 1024  # 批次最大ID位元組大小 (100KB，安全起見)
        
        # ChromaDB Cloud 配額限制（根據錯誤訊息調整）
        CHROMADB_QUOTA_LIMIT = self.quota_limit  # 使用實例變數
        SAFETY_MARGIN = 0.8  # 安全邊際，只使用 80% 的配額
        
        print(f"   - Found {len(documents_to_add)} new terms. Generating rich embeddings...")
        
        current_batch_docs, current_batch_metas, current_batch_ids = [], [], []
        current_batch_bytes = 0
        total_processed = 0
        
        for i in range(len(documents_to_add)):
            doc, meta, doc_id = documents_to_add[i], metadatas_to_add[i], ids_to_add[i]
            id_bytes = len(doc_id.encode('utf-8'))
            
            # 檢查單個 ID 是否超過配額限制
            if id_bytes > CHROMADB_QUOTA_LIMIT * SAFETY_MARGIN:
                print(f"     - ⚠️ Skipping item with ID size {id_bytes} bytes (exceeds quota limit)")
                continue

            # 檢查加入這個新項目後，是否會超過配額限制
            if (current_batch_docs and 
               (len(current_batch_docs) >= MAX_BATCH_ITEMS or 
                current_batch_bytes + id_bytes > CHROMADB_QUOTA_LIMIT * SAFETY_MARGIN)):
                
                # 提交當前的批次
                if not self._submit_batch(current_batch_docs, current_batch_metas, current_batch_ids):
                    print(f"   - ❌ Batch submission failed. Stopping processing for this file.")
                    return # 如果提交失敗（例如因為配額），則終止此檔案的處理
                
                total_processed += len(current_batch_docs)
                # 重置批次
                current_batch_docs, current_batch_metas, current_batch_ids = [], [], []
                current_batch_bytes = 0

            # 將當前項目加入到批次中
            current_batch_docs.append(doc)
            current_batch_metas.append(meta)
            current_batch_ids.append(doc_id)
            current_batch_bytes += id_bytes
        
        # 提交最後剩餘的批次
        if current_batch_docs:
            if self._submit_batch(current_batch_docs, current_batch_metas, current_batch_ids):
                total_processed += len(current_batch_docs)

        print(f"✅ Successfully processed {total_processed} terms from {file_path} (skipped {len(documents_to_add) - total_processed} due to quota limits).")
    
    def _submit_batch(self, docs, metas, ids) -> bool:
        """Helper function to submit a single batch and handle errors. Returns True on success."""
        try:
            batch_size_bytes = sum(len(i.encode('utf-8')) for i in ids)
            print(f"     - Submitting batch with {len(docs)} items ({batch_size_bytes} bytes)...")
            
            # 檢查批次大小是否超過配額
            if batch_size_bytes > self.quota_limit:  # 使用實例變數
                print(f"     - ⚠️ Batch size ({batch_size_bytes} bytes) exceeds quota limit ({self.quota_limit} bytes)")
                print(f"     - Reducing batch size and retrying...")
                
                # 動態減少批次大小
                if len(docs) > 1:
                    # 分成兩個較小的批次
                    mid = len(docs) // 2
                    batch1_success = self._submit_batch(docs[:mid], metas[:mid], ids[:mid])
                    batch2_success = self._submit_batch(docs[mid:], metas[mid:], ids[mid:])
                    return batch1_success and batch2_success
                else:
                    print(f"     - ❌ Single item too large ({batch_size_bytes} bytes). Skipping.")
                    return False
            
            self.collection.add(documents=docs, metadatas=metas, ids=ids)
            print(f"     - ✅ Batch added successfully.")
            self.processed_ids.update(ids)
            return True
            
        except chromadb.errors.ChromaError as e:
            if "Quota exceeded" in str(e):
                print(f"\n   - ❌ ChromaDB Cloud quota exceeded during batch submission.")
                print(f"   - Details: {e}")
                # 嘗試減少批次大小並重試
                if len(docs) > 1:
                    print(f"   - 🔄 Attempting to split batch and retry...")
                    mid = len(docs) // 2
                    batch1_success = self._submit_batch(docs[:mid], metas[:mid], ids[:mid])
                    batch2_success = self._submit_batch(docs[mid:], metas[mid:], ids[mid:])
                    return batch1_success and batch2_success
                else:
                    print(f"   - ❌ Single item exceeds quota. Skipping.")
                    return False
            else:
                print(f"   - ❌ A ChromaDB error occurred: {e}")
                return False
        except Exception as e:
            print(f"   - ❌ An unexpected error occurred during batch add: {e}")
            return False

    def search(self, query_text: str, n_results: int = 5) -> List[Dict]:
        """
        Searches the vector database for the most similar terms.
        """
        print(f"\n🔍 Searching for terms similar to: '{query_text}'...")
        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results
        )

        # 清理並回傳結果
        cleaned_results = []
        if results and results['metadatas'][0]:
            for i, metadata in enumerate(results['metadatas'][0]):
                cleaned_results.append({
                    "label": metadata.get("label"),
                    "uri": metadata.get("uri"),
                    "distance": results['distances'][0][i]  # 距離越小越相似
                })

        return cleaned_results


def main():
    """主函數，用於展示整個流程"""

    # --- 步驟 2.1: 建立向量資料庫 ---
    processor = VocabularyProcessor()

    # --- 這是主要的修改之處 ---
    # 自動掃描 'vocabularies' 資料夾下的所有 .ttl 和 .rdf 檔案
    # 修复路径问题：确保无论从哪个目录运行都能找到正确的vocabularies文件夹
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)  # 从src目录回到项目根目录
    vocab_directory = os.path.join(project_root, "vocabularies")

    ttl_files_to_process = []
    if not os.path.exists(vocab_directory):
        print(f"❌ 錯誤：找不到 '{vocab_directory}' 資料夾。請先建立它並放入詞彙庫檔案。")
        return

    for root, _, files in os.walk(vocab_directory):
        for file in files:
            if file.endswith((".ttl", ".rdf", ".owl")):
                full_path = os.path.join(root, file)
                ttl_files_to_process.append(full_path)

    if not ttl_files_to_process:
        print(f"⚠️ 警告：在 '{vocab_directory}' 資料夾中沒有找到任何詞彙庫檔案。")
        # 即使沒有找到檔案，我們依然可以繼續執行搜索，只是知識庫是空的
    else:
        print(f"📂 發現 {len(ttl_files_to_process)} 個詞彙庫檔案，準備處理...")

        for ttl_file in ttl_files_to_process:
            processor.process_turtle_file(ttl_file)
    # --- 修改結束 ---

    # --- 步驟 2.2: 設計檢索邏輯 (維持不變) ---

    # 模擬當用戶選擇了 'CERT' 欄位
    column_name = "CERT"
    sample_data = [14761.0, 57899.0]
    query = f"Column Name: {column_name}\nDescription: A certificate number for a financial institution.\nSample Values: {sample_data}"

    search_results = processor.search(query, n_results=5)

    print("\n📊 Search Results:")
    if search_results:
        for result in search_results:
            print(
                f"  - Label: {result['label']}\n    URI: {result['uri']}\n    Distance: {result['distance']:.4f}\n")
    else:
        print("   - No results found.")


if __name__ == "__main__":
    main()
