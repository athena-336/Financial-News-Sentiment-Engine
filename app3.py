# app.py
from flask import Flask, jsonify, request, render_template
from pymongo import MongoClient
from datetime import datetime, timedelta

from flask_cors import CORS
app = Flask(__name__)
CORS(app)


# -------------------------
# 1. MongoDB Connection
# -------------------------
client = MongoClient("mongodb://localhost:27017")
db = client["sentiment"]
collection = db["articles_new"]


# -------------------------
# 2. 工具函数
# -------------------------
def parse_date(date_str, default=None):
    """把 YYYY-MM-DD 字符串转换为 datetime"""
    if not date_str:
        return default
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except:
        return default


def sentiment_label(score, pos_th=0.2, neg_th=-0.2):
    """根据 sentiment_score 给 Positive/Neutral/Negative 标签"""
    if score is None:
        return "Neutral"
    if score > pos_th:
        return "Positive"
    if score < neg_th:
        return "Negative"
    return "Neutral"


# -------------------------
# 3. 首页：渲染 web.html
# -------------------------
@app.route("/")
def index():
    return render_template("web.html")


# -------------------------
# 4. API：公司总览
# -------------------------
@app.route("/api/company_summary")
def company_summary():
    pipeline = [
        {
            "$group": {
                "_id": "$company",
                "article_count": {"$sum": 1},
                "avg_sentiment": {"$avg": "$sentiment_score"},
                "first_date": {"$min": "$published_at"},
                "last_date": {"$max": "$published_at"},
            }
        },
        {"$sort": {"_id": 1}}
    ]

    result = list(collection.aggregate(pipeline))

    data = []
    for r in result:
        data.append({
            "company": r["_id"],
            "article_count": r["article_count"],
            "avg_sentiment": round(r["avg_sentiment"], 4) if r.get("avg_sentiment") is not None else None,
            "first_date": r["first_date"].strftime("%Y-%m-%d") if r.get("first_date") else None,
            "last_date": r["last_date"].strftime("%Y-%m-%d") if r.get("last_date") else None,
        })

    return jsonify(data)


# -------------------------
# 5. API：文章列表（分页+筛选）
# -------------------------
@app.route("/api/articles")
def get_articles():
    company = request.args.get("company")
    if not company:
        return jsonify({"error": "company is required"}), 400

    start = parse_date(request.args.get("start"))
    end = parse_date(request.args.get("end"))
    label = request.args.get("label", "all").lower()
    sort = request.args.get("sort", "newest").lower()

    try:
        page = int(request.args.get("page", 1))
        limit = int(request.args.get("limit", 20))
    except:
        page, limit = 1, 20

    query = {"company": company}

    # 日期条件
    if start or end:
        query["published_at"] = {}
        if start:
            query["published_at"]["$gte"] = start
        if end:
            query["published_at"]["$lte"] = end + timedelta(days=1)

    # 情绪筛选
    if label == "positive":
        query["sentiment_score"] = {"$gt": 0.2}
    elif label == "negative":
        query["sentiment_score"] = {"$lt": -0.2}
    elif label == "neutral":
        query["sentiment_score"] = {"$gte": -0.2, "$lte": 0.2}

    # 排序
    if sort == "oldest":
        sort_spec = [("published_at", 1)]
    elif sort == "high_sent":
        sort_spec = [("sentiment_score", -1)]
    elif sort == "low_sent":
        sort_spec = [("sentiment_score", 1)]
    else:  # newest
        sort_spec = [("published_at", -1)]

    skip = (page - 1) * limit

    cursor = collection.find(query).sort(sort_spec).skip(skip).limit(limit)

    items = []
    for doc in cursor:
        items.append({
            "title": doc.get("title"),
            "description": doc.get("description"),
            "content": doc.get("content"),
            "source": doc.get("source"),
            "url": doc.get("url"),
            "company": doc.get("company"),
            "published_at": doc.get("published_at").strftime("%Y-%m-%d") if doc.get("published_at") else None,
            "sentiment_score": doc.get("sentiment_score"),
            "sentiment_label": sentiment_label(doc.get("sentiment_score")),
        })

    return jsonify({
        "page": page,
        "limit": limit,
        "count": len(items),
        "items": items
    })


# -------------------------
# 6. API：公司统计 + 饼图 + 时间序列
# -------------------------
@app.route("/api/company_stats")
def company_stats():
    company = request.args.get("company")
    if not company:
        return jsonify({"error": "company is required"}), 400

    start = parse_date(request.args.get("start"))
    end = parse_date(request.args.get("end"))

    match_stage = {"company": company}

    if start or end:
        match_stage["published_at"] = {}
        if start:
            match_stage["published_at"]["$gte"] = start
        if end:
            match_stage["published_at"]["$lte"] = end + timedelta(days=1)

    # ---- overall ----
    pipeline_overall = [
        {"$match": match_stage},
        {
            "$group": {
                "_id": None,
                "article_count": {"$sum": 1},
                "avg_sentiment": {"$avg": "$sentiment_score"},
            }
        }
    ]

    overall_result = list(collection.aggregate(pipeline_overall))
    if overall_result:
        res = overall_result[0]
        overall = {
            "article_count": res["article_count"],
            "avg_sentiment": round(res["avg_sentiment"], 4) if res["avg_sentiment"] is not None else None
        }
    else:
        overall = {"article_count": 0, "avg_sentiment": None}

    # ---- breakdown ----
    pipeline_breakdown = [
        {"$match": match_stage},
        {
            "$project": {
                "label": {
                    "$switch": {
                        "branches": [
                            {"case": {"$gt": ["$sentiment_score", 0.2]}, "then": "Positive"},
                            {"case": {"$lt": ["$sentiment_score", -0.2]}, "then": "Negative"},
                        ],
                        "default": "Neutral"
                    }
                }
            }
        },
        {"$group": {"_id": "$label", "count": {"$sum": 1}}},
    ]

    breakdown_raw = list(collection.aggregate(pipeline_breakdown))
    breakdown = {"Positive": 0, "Neutral": 0, "Negative": 0}
    for b in breakdown_raw:
        breakdown[b["_id"]] = b["count"]

    # ---- time series ----
    pipeline_ts = [
        {"$match": match_stage},
        {
            "$group": {
                "_id": {
                    "$dateToString": {"format": "%Y-%m-%d", "date": "$published_at"}
                },
                "article_count": {"$sum": 1},
                "avg_sentiment": {"$avg": "$sentiment_score"},
            }
        },
        {"$sort": {"_id": 1}}
    ]

    ts_raw = list(collection.aggregate(pipeline_ts))
    time_series = []
    for t in ts_raw:
        time_series.append({
            "date": t["_id"],
            "article_count": t["article_count"],
            "avg_sentiment": round(t["avg_sentiment"], 4),
        })

    return jsonify({
        "company": company,
        "overall": overall,
        "breakdown": breakdown,
        "time_series": time_series
    })


# -------------------------
# 7. 主入口：port=5001
# -------------------------
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=True)
