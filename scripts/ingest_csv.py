#!/usr/bin/env python3
"""
Импорт товаров из CSV файла российских супермаркетов.

Использование:
    uv run python scripts/ingest_csv.py [--limit N] [--reset]

Аргументы:
    --limit N    Количество товаров для загрузки (по умолчанию 1000)
    --reset      Сбросить коллекцию перед импортом
"""

import argparse
import csv
import hashlib
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Загружаем .env перед импортом settings (override=True чтобы перезаписать env)
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path, override=True)
print(f"EMBEDDING_MODEL: {os.environ.get('EMBEDDING_MODEL')}")

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.vector_store import VectorStore


class ProductData:
    def __init__(self, product_id: str, name: str, category: str, price: float, brand: str = "", manufacturer: str = "", package_size: str = ""):
        self.product_id = product_id
        self.name = name
        self.category = category
        self.price = price
        self.brand = brand
        self.manufacturer = manufacturer
        self.package_size = package_size

    def to_document_text(self) -> str:
        parts = [
            f"Название: {self.name}",
            f"Категория: {self.category}",
            f"Цена: {self.price} руб.",
        ]
        if self.brand:
            parts.append(f"Бренд: {self.brand}")
        if self.manufacturer:
            parts.append(f"Производитель: {self.manufacturer}")
        if self.package_size:
            parts.append(f"Упаковка: {self.package_size}")
        return " | ".join(parts)

    def to_vector_dict(self) -> dict:
        return {
            "id": self.get_id(),
            "text": self.to_document_text(),
            "metadata": self.get_metadata()
        }

    def get_metadata(self) -> dict:
        return {
            "product_id": self.product_id,
            "name": self.name,
            "category": self.category,
            "price": str(self.price),
            "brand": self.brand,
            "manufacturer": self.manufacturer,
            "package_size": self.package_size,
        }

    def get_id(self) -> str:
        return hashlib.md5(f"product_{self.product_id}".encode()).hexdigest()


def parse_price(price_str: str) -> float:
    """Распарсить цену из строки"""
    if not price_str:
        return 0.0
    try:
        return float(price_str.replace(',', '.').strip())
    except ValueError:
        return 0.0


def clean_text(text: str) -> str:
    """Очистить текст от лишних пробелов"""
    if not text:
        return ""
    return text.strip()


def ingest_csv(file_path: str, limit: int = 1000, reset: bool = False):
    """Импортировать товары из CSV"""
    print("=" * 50)
    print(f"Импорт из {file_path}")
    print("=" * 50)

    vector_store = VectorStore(collection_name="products")

    if reset:
        print("Сброс существующей коллекции...")
        vector_store.reset()

    products_data = []
    row_count = 0

    print(f"\nЧтение CSV (лимит: {limit if limit else 'все'})...")

    with open(file_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if limit and len(products_data) >= limit:
                break

            name = clean_text(row.get('product_name', ''))
            category = clean_text(row.get('product_category', ''))
            brand = clean_text(row.get('brand', ''))
            manufacturer = clean_text(row.get('manufacturer', ''))
            package_size = clean_text(row.get('package_size', ''))
            price = parse_price(row.get('new_price', '0'))

            # Пропускаем пустые строки
            if not name:
                continue

            # Если цены нет, ставим среднюю
            if price <= 0:
                price = 100.0

            product = ProductData(
                product_id=f"csv_{row_count}",
                name=name,
                category=category if category else "Разное",
                price=price,
                brand=brand,
                manufacturer=manufacturer,
                package_size=package_size
            )
            products_data.append(product.to_vector_dict())
            row_count += 1

            if row_count % 100 == 0:
                print(f"  Обработано: {row_count}")

    print(f"\nПодготовлено документов: {len(products_data)}")

    print("\nСохранение в векторную базу данных...")
    vector_store.add_products(products_data)

    stats = vector_store.get_stats()
    print("\n" + "=" * 50)
    print("Импорт завершён")
    print(f"Всего документов в БД: {stats['total_documents']}")
    print("=" * 50)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Импорт товаров из CSV"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Количество товаров для загрузки (None = все)"
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Сбросить коллекцию перед импортом"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    csv_path = "data/russian_supermarket_prices.csv"
    ingest_csv(csv_path, limit=args.limit, reset=args.reset)


if __name__ == "__main__":
    main()
