import streamlit as st
import pandas as pd
import joblib

# تحميل الموديلات والأدوات من الملف المضغوط
artifacts = joblib.load('crop_model.pkl')

reg_model   = artifacts['reg_model']
reg_le      = artifacts['reg_le']
reg_scaler  = artifacts['reg_scaler']
reg_columns = artifacts['reg_columns']

clf_model   = artifacts['clf_model']
clf_le      = artifacts['clf_le']
clf_scaler  = artifacts['clf_scaler']
clf_columns = artifacts['clf_columns']

st.set_page_config(page_title="FAOSTAT Egypt Agriculture AI", page_icon="🌾")
st.title("🌾 نظام التنبؤ بالإنتاج والمخاطر الزراعية (FAOSTAT)")

tab1, tab2 = st.tabs(["📊 توقع الإنتاج (Regression)", "⚠️ تقييم خطر الانخفاض (Classification)"])

# ==========================================
# التاب الأول: Regression (توقع الإنتاج/الإنتاجية)
# ==========================================
with tab1:
    st.subheader("توقع كمية الإنتاج أو الإنتاجية")
    crop_reg = st.selectbox("اختر المحصول (Item):", reg_le.classes_, key="reg_crop")
    year_reg = st.number_input("السنة (Year):", min_value=1990, max_value=2035, value=2025, key="reg_year")
    element_type = st.radio("نوع القياس (Element):", ["Production (طن)", "Yield (كجم/هكتار)", "Area harvested (هكتار)"])

    if st.button("توقع القيمة", key="btn_reg"):
        input_reg = pd.DataFrame(0.0, index=[0], columns=reg_columns)
        input_reg['Item'] = reg_le.transform([crop_reg])[0]
        input_reg['Year'] = reg_scaler.transform(pd.DataFrame({'Year': [year_reg]}))[0][0]

        if "Production" in element_type and 'Element_Production' in input_reg.columns:
            input_reg['Element_Production'] = 1
        elif "Yield" in element_type and 'Element_Yield' in input_reg.columns:
            input_reg['Element_Yield'] = 1
            
        if 'Flag_E' in input_reg.columns:
            input_reg['Flag_E'] = 1

        pred_value = reg_model.predict(input_reg)[0]
        st.success(f"القيمة المتوقعة لـ **{crop_reg}** في سنة **{year_reg}**: **{pred_value:,.2f}**")

# ==========================================
# التاب الثاني: Classification (توقع خطر الانخفاض)
# ==========================================
with tab2:
    st.subheader("تقييم احتمالية انخفاض المحصول (Safe vs Risk)")
    crop_clf = st.selectbox("اختر المحصول (Item):", clf_le.classes_, key="clf_crop")
    year_clf = st.number_input("السنة (Year):", min_value=1991, max_value=2035, value=2025, key="clf_year")
    prev_val = st.number_input("إنتاج السنة السابقة بالطن (Prev_Year_Value):", min_value=0.0, value=100000.0, step=1000.0)
    flag_type = st.selectbox("مصدر البيانات (Flag):", ["E (Estimated)", "I (Imputed)", "X (International)", "A (Official)"])

    if st.button("تحليل الخطر", key="btn_clf"):
        input_clf = pd.DataFrame(0.0, index=[0], columns=clf_columns)
        input_clf['Item'] = clf_le.transform([crop_clf])[0]

        # عمل Scaling بنفس ترتيب الأعمدة في النوت بوك: ['Year', 'Prev_Year_Value']
        num_df = pd.DataFrame({'Year': [year_clf], 'Prev_Year_Value': [prev_val]})
        scaled_vals = clf_scaler.transform(num_df)[0]
        input_clf['Year'] = scaled_vals[0]
        input_clf['Prev_Year_Value'] = scaled_vals[1]

        # تحديد الـ Flag المختار
        flag_code = f"Flag_{flag_type[0]}"
        if flag_code in input_clf.columns:
            input_clf[flag_code] = 1

        risk_pred = clf_model.predict(input_clf)[0]
        risk_prob = clf_model.predict_proba(input_clf)[0]

        if risk_pred == 1:
            st.error(f"⚠️ **تحذير (Risk = 1):** متوقع انخفاض في إنتاج **{crop_clf}** عن المتوسط التاريخي بنسبة احتمال **{risk_prob[1]*100:.1f}%**")
        else:
            st.success(f"✅ **وضع آمن (Safe = 0):** إنتاج **{crop_clf}** مستقر أو أعلى من المتوسط بنسبة احتمال **{risk_prob[0]*100:.1f}%**")