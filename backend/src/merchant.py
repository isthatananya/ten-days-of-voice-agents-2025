import json
import os
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional

# Simple product catalog (ACP-inspired minimal schema)
PRODUCTS: List[Dict[str, Any]] = [
    {
        "id": "mug-001",
        "name": "Stoneware Coffee Mug",
        "description": "12oz white stoneware mug",
        "price": 800,
        "currency": "INR",
        "category": "mug",
        "color": "white",
        "sizes": [],
    },
    {
        "id": "mug-002",
        "name": "Blue Ceramic Mug",
        "description": "12oz ceramic mug in deep blue",
        "price": 850,
        "currency": "INR",
        "category": "mug",
        "color": "blue",
        "sizes": [],
    },
    {
        "id": "tee-001",
        "name": "Classic Tee",
        "description": "Cotton t-shirt, unisex",
        "price": 699,
        "currency": "INR",
        "category": "tshirt",
        "color": "black",
        "sizes": ["S", "M", "L", "XL"],
    },
    {
        "id": "hoodie-001",
        "name": "Pullover Hoodie",
        "description": "Warm fleece hoodie",
        "price": 1499,
        "currency": "INR",
        "category": "hoodie",
        "color": "black",
        "sizes": ["S", "M", "L", "XL"],
    },
    {
        "id": "hoodie-002",
        "name": "Zip Hoodie",
        "description": "Full-zip hoodie, gray",
        "price": 1699,
        "currency": "INR",
        "category": "hoodie",
        "color": "gray",
        "sizes": ["S", "M", "L", "XL"],
    },
    {
        "id": "cap-001",
        "name": "Baseball Cap",
        "description": "Adjustable cap",
        "price": 399,
        "currency": "INR",
        "category": "cap",
        "color": "navy",
        "sizes": [],
    },
]

ORDERS: List[Dict[str, Any]] = []
ORDERS_FILE = "orders.json"


def _persist_orders() -> None:
    try:
        with open(ORDERS_FILE, "w", encoding="utf-8") as f:
            json.dump(ORDERS, f, indent=2, ensure_ascii=False)
    except Exception:
        # Non-fatal — best-effort persistence
        pass


def list_products(filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Return products matching simple filters.

    Supported filters (naive):
    - category: str
    - max_price: number
    - color: str
    - size: str
    - text: substring match in name/description
    """
    results = PRODUCTS
    if not filters:
        return results

    f = filters
    if f.get("category"):
        results = [p for p in results if p.get("category") == f["category"]]
    if f.get("max_price") is not None:
        try:
            maxp = float(f["max_price"])
            results = [p for p in results if float(p.get("price", 0)) <= maxp]
        except Exception:
            pass
    if f.get("color"):
        color = f["color"].lower()
        results = [p for p in results if p.get("color", "").lower() == color]
    if f.get("size"):
        size = str(f["size"]).upper()
        results = [p for p in results if size in [s.upper() for s in p.get("sizes", [])]]
    if f.get("text"):
        q = f["text"].lower()
        results = [p for p in results if q in p.get("name", "").lower() or q in p.get("description", "").lower()]

    return results


def create_order(line_items: List[Dict[str, Any]], metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Create an order from line_items: [{product_id, quantity, attributes?}]

    Computes a naive total and appends the order to ORDERS and persists to `orders.json`.
    """
    items = []
    total = 0.0
    currency = "INR"

    for li in line_items:
        pid = li.get("product_id")
        qty = int(li.get("quantity", 1))
        product = next((p for p in PRODUCTS if p["id"] == pid), None)
        if not product:
            raise ValueError(f"Unknown product id: {pid}")
        price = float(product.get("price", 0))
        line_total = price * qty
        total += line_total
        currency = product.get("currency", currency)
        items.append({
            "product_id": pid,
            "name": product.get("name"),
            "unit_price": price,
            "quantity": qty,
            "line_total": line_total,
            "attributes": li.get("attributes", {}),
        })

    order = {
        "id": str(uuid.uuid4()),
        "items": items,
        "total": total,
        "currency": currency,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "metadata": metadata or {},
    }

    ORDERS.append(order)
    _persist_orders()
    return order


def get_last_order() -> Optional[Dict[str, Any]]:
    if not ORDERS:
        return None
    return ORDERS[-1]


# Try to load persisted orders at import-time (best-effort)
if os.path.exists(ORDERS_FILE):
    try:
        with open(ORDERS_FILE, "r", encoding="utf-8") as f:
            ORDERS = json.load(f)
    except Exception:
        ORDERS = []
