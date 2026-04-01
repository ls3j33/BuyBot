import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2
from typing import List
from src.config.settings import settings


class HuggingFaceEmbeddingFunction:
    """Кастомная embedding функция для поддержки trust_remote_code"""

    def __init__(self, model_name: str, load_in_8bit: bool = False, load_in_4bit: bool = False):
        from sentence_transformers import SentenceTransformer
        # Используем sentence-transformers напрямую для лучшей совместимости
        self.model = SentenceTransformer(
            model_name,
            trust_remote_code=True,
        )
        self._model_name = model_name

    def __call__(self, input: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(input, normalize_embeddings=True)
        return embeddings.tolist()

    def embed_query(self, input: str) -> List[float]:
        """Embed a single query text"""
        # ChromaDB может передавать как строку, так и список
        if isinstance(input, list):
            input = input[0] if input else ""
        embedding = self.model.encode(input, normalize_embeddings=True)
        return embedding.tolist()

    def embed_documents(self, input: List[str]) -> List[List[float]]:
        """Embed multiple documents"""
        embeddings = self.model.encode(input, normalize_embeddings=True)
        return embeddings.tolist()

    def name(self) -> str:
        """Return the name of the embedding function for ChromaDB"""
        return f"HuggingFaceEmbeddingFunction({self._model_name})"


class VectorStore:
    """Класс для работы с векторной базой данных ChromaDB"""

    def __init__(self, collection_name: str = "products"):
        """
        Инициализация ChromaDB.

        Args:
            collection_name: Имя коллекции для хранения продуктов
        """
        self.client = chromadb.PersistentClient(
            path=settings.chroma_db_path,
            settings=ChromaSettings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        # Используем ту же модель, что и в settings
        self.embeddings = HuggingFaceEmbeddingFunction(
            model_name=settings.embedding_model,
        )
        self.collection_name = collection_name
        self._collection = None
    
    @property
    def collection(self):
        """Ленивое получение коллекции"""
        if self._collection is None:
            # Проверяем существует ли коллекция
            existing_collections = self.client.list_collections()
            collection_exists = any(coll.name == self.collection_name for coll in existing_collections)

            if collection_exists:
                # Получаем существующую коллекцию с embedding_function для правильной генерации эмбеддингов
                self._collection = self.client.get_collection(name=self.collection_name, embedding_function=self.embeddings)
            else:
                # Создаём новую коллекцию
                self._collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"},  # Косинусное расстояние
                    embedding_function=self.embeddings
                )
        return self._collection
    
    def add_products(self, products: list[dict]) -> None:
        """
        Добавить продукты в векторную базу данных.

        Args:
            products: Список словарей с данными продуктов:
                - id: ID продукта
                - text: Текст для векторизации
                - metadata: Метаданные
        """
        if not products:
            print("Нет продуктов для добавления")
            return

        # Подготовка данных для ChromaDB
        ids = []
        documents = []
        metadatas = []

        for product in products:
            doc_id = product.get("id")

            # Проверка на дубликаты
            existing = self.collection.get(ids=[doc_id])
            if existing["ids"]:
                print(f"Продукт {product.get('name', doc_id)} уже существует, пропускаем")
                continue

            ids.append(doc_id)
            documents.append(product["text"])
            metadatas.append(product.get("metadata", {}))

        if ids:
            # ChromaDB имеет лимит на размер пакета (~5461)
            # Разбиваем на пакеты по 5000
            batch_size = 5000
            total_added = 0
            
            for i in range(0, len(ids), batch_size):
                batch_ids = ids[i:i + batch_size]
                batch_docs = documents[i:i + batch_size]
                batch_meta = metadatas[i:i + batch_size]
                
                self.collection.add(
                    ids=batch_ids,
                    documents=batch_docs,
                    metadatas=batch_meta
                )
                total_added += len(batch_ids)
                print(f"  Добавлено {len(batch_ids)} продуктов (всего: {total_added})")
            
            print(f"Добавлено {total_added} продуктов в базу данных")
        else:
            print("Все продукты уже существуют в базе данных")
    
    def search(
        self,
        query: str,
        n_results: int = 5,
        filter_category: str | None = None
    ) -> list[dict]:
        """
        Поиск релевантных продуктов.

        Args:
            query: Текст запроса для поиска
            n_results: Количество результатов для возврата
            filter_category: Опциональный фильтр по категории

        Returns:
            Список найденных документов с метаданными
        """
        where_clause = None
        if filter_category:
            where_clause = {"category": filter_category}

        # Генерируем эмбеддинг вручную для обхода бага ChromaDB 1.5.5 с query_texts
        query_embedding = self.embeddings.embed_query(query)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_clause,
            include=["documents", "metadatas", "distances"]
        )
        
        # Преобразование результатов в удобный формат
        formatted_results = []
        if results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                formatted_results.append({
                    "id": doc_id,
                    "document": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i] if results["distances"] else None,
                })
        
        return formatted_results
    
    def get_stats(self) -> dict:
        """Получить статистику по коллекции"""
        return {
            "collection_name": self.collection_name,
            "total_documents": self.collection.count(),
        }
    
    def reset(self) -> None:
        """Очистить коллекцию (для отладки)"""
        try:
            self.client.delete_collection(self.collection_name)
            self._collection = None
            print(f"Коллекция {self.collection_name} удалена")
        except Exception as e:
            print(f"Ошибка при удалении коллекции: {e}")
