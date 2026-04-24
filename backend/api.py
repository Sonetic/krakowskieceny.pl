from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import io
import zipfile
import stripe
import os
import sys
from supabase import create_client
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
from predykcja import predict_price
from datetime import datetime, timedelta, timezone
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address


# =========================
# APP
# =========================
app = Flask(__name__)

CORS(
    app,
    resources={r"/*": {"origins": [
            "https://krakowskieceny-pl.onrender.com"
        ]}}
)

# =========================
# ENV
# =========================
stripe.api_key = os.getenv("STRIPE_SECRET_KEY")
WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET")

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

PRICE = 200  # grosze

# =========================
# CREATE PAYMENT SESSION
# =========================
@app.route("/create-checkout-session", methods=["POST"])
def create_checkout_session():
    data = request.get_json()

    session = stripe.checkout.Session.create(
        mode="payment",
        payment_method_types=["card"],
        line_items=[
            {
                "price_data": {
                    "currency": "pln",
                    "product_data": {
                        "name": "Raport cen mieszkań Warszawa",
                    },
                    "unit_amount": PRICE,
                },
                "quantity": 1,
            }
        ],
        success_url="https://krakowskieceny-pl.onrender.com/success.html?session_id={CHECKOUT_SESSION_ID}",
        cancel_url="https://krakowskieceny-pl.onrender.com/predykcja.html",
        metadata={
            "ulica": data.get("ulica"),
            "numer": data.get("numer"),
        }
    )

    return jsonify({"url": session.url})


# =========================
# STRIPE WEBHOOK
# =========================
@app.route("/webhook", methods=["POST"])
def webhook():
    payload = request.data
    sig_header = request.headers.get("Stripe-Signature")

    try:
        event = stripe.Webhook.construct_event(
            payload,
            sig_header,
            WEBHOOK_SECRET
        )
    except Exception as e:
        return str(e), 400

    # PAYMENT SUCCESS
    if event["type"] == "checkout.session.completed":
        session = event["data"]["object"]
        session_id = session["id"]

        supabase.table("payments").upsert({
            "id": session_id,
            "paid": True,
            "used": False
        }).execute()

        print("PAYMENT SAVED:", session_id)

    return "ok", 200


# =========================
# PREDICTION (PROTECTED)
# =========================

limiter = Limiter(get_remote_address, app=app)

@app.route("/predict", methods=["POST"])
@limiter.limit("7 per minute")
def predict():
    data = request.get_json()
    session_id = data.get("session_id")

    if not session_id:
        return jsonify({"error": "Brak session_id"}), 400

    # CHECK SUPABASE
    res = supabase.table("payments").select("*").eq("id", session_id).execute()

    if len(res.data) == 0:
        return jsonify({"error": "Brak płatności"}), 403

    payment = res.data[0]

    if not payment["paid"]:
        return jsonify({"error": "Nieopłacone"}), 402

    now = datetime.now(timezone.utc)

    if not payment["used"]:
        supabase.table("payments").update({
            "used": True,
            "expires_at": (now + timedelta(hours=1)).isoformat()
        }).eq("id", session_id).execute()
    else:
        expires_at = payment.get("expires_at")

        if not expires_at:
            return jsonify({"error": "Błąd danych"}), 500

        expires_at_dt = datetime.fromisoformat(expires_at)

        if expires_at_dt.tzinfo is None:
            expires_at_dt = expires_at_dt.replace(tzinfo=timezone.utc)

        if now > expires_at_dt:
            return jsonify({"error": "Dostęp wygasł"}), 403

    # =========================
    # ML LOGIC
    # =========================
    result = predict_price(
        data["ulica"],
        data["numer"],
        data["powierzchnia"],
        data["pietro"],
        data["liczba_pokoi"]
    )

    # =========================
    # CSV OUTPUT
    # =========================
    pred_csv = io.StringIO()
    pd.DataFrame([result[0]]).to_csv(pred_csv, index=False)

    okolica_csv = io.StringIO()
    result[1].to_csv(okolica_csv, index=False)

    ulica_csv = None
    budynek_csv = None

    if len(result) > 2:
        ulica_csv = io.StringIO()
        result[2].to_csv(ulica_csv, index=False)

    if len(result) > 3:
        budynek_csv = io.StringIO()
        result[3].to_csv(budynek_csv, index=False)

    # =========================
    # ZIP RESPONSE
    # =========================
    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, "w") as zf:
        zf.writestr("predykcja.csv", pred_csv.getvalue())
        zf.writestr("okolica.csv", okolica_csv.getvalue())

        if ulica_csv:
            zf.writestr("ulica.csv", ulica_csv.getvalue())

        if budynek_csv:
            zf.writestr("budynek.csv", budynek_csv.getvalue())

    zip_buffer.seek(0)

    return send_file(
        zip_buffer,
        mimetype="application/zip",
        as_attachment=True,
        download_name="wyniki.zip"
    )


# =========================
# START
# =========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)