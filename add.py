import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np
import sklearn.compose._column_transformer as ct
from sklearn.preprocessing import FunctionTransformer
from io import BytesIO

# ===============================
# 🔧 تعريف extract_date_features كما في التدريب
# ===============================
def extract_date_features(df):
    df = df.copy()
    if "Date_of_Journey" in df.columns:
        df["Date_of_Journey"] = pd.to_datetime(df["Date_of_Journey"], dayfirst=True)
        df["Journey_Day"] = df["Date_of_Journey"].dt.day
        df["Journey_Month"] = df["Date_of_Journey"].dt.month
        df["Journey_Year"] = df["Date_of_Journey"].dt.year
        df.drop("Date_of_Journey", axis=1, inplace=True)
    return df

# ===============================
# 🧩 إصلاح مشكلة _RemainderColsList
# ===============================
class _RemainderColsList(list):
    """Dummy class used to patch sklearn deserialization"""
    pass

setattr(ct, "_RemainderColsList", _RemainderColsList)

# ===============================
# إعداد الصفحة
# ===============================
st.set_page_config(page_title="تطبيق تسعير الطيران", page_icon="✈️", layout="wide")
st.title("✈️ تطبيق تسعير تذاكر الطيران")
st.markdown("أدخل بيانات الرحلة لتوقع سعر التذكرة بناءً على الموديل المدرب 🎯")

# ===============================
# تحميل الموديل والبيانات
# ===============================
MODEL_PATH = os.path.join(os.getcwd(), "best_model1.pkl")
DATA_PATH = os.path.join(os.getcwd(), "Data_Train.xlsx")

try:
    globals()["extract_date_features"] = extract_date_features
    globals()["_RemainderColsList"] = _RemainderColsList
    globals()["FunctionTransformer"] = FunctionTransformer

    model = joblib.load(MODEL_PATH)
    st.sidebar.success("✅ تم تحميل الموديل بنجاح")
except Exception as e:
    st.sidebar.error(f"❌ خطأ في تحميل الموديل: {e}")
    model = None

try:
    data = pd.read_excel(DATA_PATH, engine="openpyxl")
    st.sidebar.success("✅ تم تحميل البيانات بنجاح")
except Exception as e:
    st.sidebar.error(f"❌ خطأ في تحميل البيانات: {e}")
    data = None

# ===============================
# تهيئة تخزين التوقعات في session_state
# ===============================
if "predictions" not in st.session_state:
    st.session_state["predictions"] = pd.DataFrame(columns=[
        "Airline", "Source", "Destination", "Total_Stops",
        "Duration_Minutes", "Days_Left", "Predicted_Price"
    ])

# ===============================
# واجهة إدخال البيانات
# ===============================
if data is not None and model is not None:
    st.header("🧾 إدخال بيانات الرحلة")

    col1, col2 = st.columns(2)
    with col1:
        airline = st.selectbox("شركة الطيران", sorted(data["Airline"].dropna().unique()))
        source = st.selectbox("مكان الإقلاع", sorted(data["Source"].dropna().unique()))
        destination = st.selectbox("الوجهة", sorted(data["Destination"].dropna().unique()))
    with col2:
        total_stops = st.selectbox("عدد التوقفات", ["non-stop", "1 stop", "2 stops", "3 stops", "4 stops"])
        duration = st.slider("⏱️ مدة الرحلة (بالساعات)", 1, 40, 5)
        days_left = st.slider("📆 عدد الأيام قبل الرحلة", 0, 60, 10)
    
    date_of_journey = st.date_input("📅 تاريخ الرحلة")
    dep_time = st.time_input("⏰ وقت الإقلاع")

    # بناء الداتا فريم
    input_df = pd.DataFrame({
        "Airline": [airline],
        "Source": [source],
        "Destination": [destination],
        "Total_Stops": [total_stops],
        "Duration": [f"{duration}h"],
        "Days_Left": [days_left],
        "Date_of_Journey": [pd.to_datetime(date_of_journey)],
        "Dep_Time": [dep_time.strftime("%H:%M")],
        "Additional_Info": ["No info"]
    })

    input_df = extract_date_features(input_df)
    input_df["Dep_Hour"] = pd.to_datetime(input_df["Dep_Time"]).dt.hour
    input_df["Dep_Minute"] = pd.to_datetime(input_df["Dep_Time"]).dt.minute
    input_df.drop("Dep_Time", axis=1, inplace=True)
    stops_map = {"non-stop": 0, "1 stop": 1, "2 stops": 2, "3 stops": 3, "4 stops": 4}
    input_df["Total_Stops"] = input_df["Total_Stops"].map(stops_map)
    input_df["Duration_Minutes"] = input_df["Duration"].str.replace("h", "").astype(float) * 60
    input_df.drop("Duration", axis=1, inplace=True)

    st.write("### 👇 البيانات المدخلة بعد المعالجة")
    st.dataframe(input_df)

    # ===============================
    # التنبؤ
    # ===============================
    if st.button("🔮 توقع السعر"):
        try:
            prediction_log = model.predict(input_df)
            prediction = np.expm1(prediction_log)
            price = float(prediction[0])

            st.success(f"💰 السعر المتوقع للتذكرة هو: {price:,.2f} جنيه")

            # حفظ النتيجة في الداشبورد
            new_row = {
                "Airline": airline,
                "Source": source,
                "Destination": destination,
                "Total_Stops": total_stops,
                "Duration_Minutes": duration * 60,
                "Days_Left": days_left,
                "Predicted_Price": price
            }
            st.session_state["predictions"] = pd.concat(
                [st.session_state["predictions"], pd.DataFrame([new_row])],
                ignore_index=True
            )

        except Exception as e:
            st.error(f"⚠️ حدث خطأ أثناء التنبؤ: {e}")

    # ===============================
    # 📊 عرض الداشبورد التفاعلية
    # ===============================
    if not st.session_state["predictions"].empty:
        st.markdown("---")
        st.subheader("📊 لوحة التوقعات السابقة")

        df_pred = st.session_state["predictions"]

        col1, col2, col3 = st.columns(3)
        col1.metric("عدد التوقعات", len(df_pred))
        col2.metric("أرخص سعر", f"{df_pred['Predicted_Price'].min():,.2f} جنيه")
        col3.metric("أعلى سعر", f"{df_pred['Predicted_Price'].max():,.2f} جنيه")

        st.dataframe(df_pred)

        st.bar_chart(df_pred.groupby("Airline")["Predicted_Price"].mean())

        # ===============================
        # 📤 تحميل النتائج و 🗑️ إعادة التعيين
        # ===============================
        st.markdown("### ⚙️ أدوات إضافية")

        colA, colB = st.columns(2)

        # 🔹 زر التحميل كملف Excel
        with colA:
            buffer = BytesIO()
            with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                df_pred.to_excel(writer, index=False, sheet_name="Predictions")
            st.download_button(
                label="📤 تحميل النتائج كملف Excel",
                data=buffer.getvalue(),
                file_name="Predicted_Flights.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

        # 🔹 زر إعادة التعيين
        with colB:
            if st.button("🗑️ إعادة تعيين التوقعات"):
                st.session_state["predictions"] = pd.DataFrame(columns=df_pred.columns)
                st.success("✅ تم مسح جميع التوقعات.")

else:
    st.error("⚠️ لم يتم تحميل البيانات أو الموديل بشكل صحيح. تأكد من وجود الملفات في نفس المجلد.")

st.markdown("---")
st.caption("🚀 تم إنشاء هذا التطبيق باستخدام Streamlit و Scikit-learn")
