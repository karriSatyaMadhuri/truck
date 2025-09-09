import streamlit as st

st.set_page_config(page_title="Maximise Truck Load", layout="wide")
st.title("📦 Maximise Truck Load")

# -------------------------------
# Inputs
# -------------------------------
box_count = st.number_input("How many box types?", min_value=1, max_value=10, value=1, key="box_count")

box_data = []
for i in range(box_count):
    st.markdown(f"### Box {i+1}")
    box_type = st.selectbox(f"Select box type for Box {i+1}", ["plc", "FLC", "CRATE", "PP"], key=f"type_{i}")
    quantity = st.number_input(f"No. of boxes (optional)", min_value=200, key=f"qty_{i}")
    length = st.number_input("External Length (mm)", min_value=100, key=f"len_{i}")
    width = st.number_input("External Width (mm)", min_value=100, key=f"wid_{i}")
    height = st.number_input("External Height (mm)", min_value=100, key=f"hei_{i}")
    payload = st.number_input("Max Payload (kg)", min_value=100, key=f"pay_{i}")

    box_data.append({
        "type": box_type,
        "quantity": quantity,
        "dimensions": (length, width, height),
        "payload": payload
    })

source = st.selectbox("Source City", ["Mumbai", "Delhi", "Chennai", "Hyderabad"], key="source_city")
destination = st.selectbox("Destination City", ["Bangalore", "Kolkata", "Pune", "Ahmedabad"], key="dest_city")

route_types = {
    "Highway": st.slider("Highway (%)", 0, 100, 50, key="highway"),
    "Semi-Urban": st.slider("Semi-Urban (%)", 0, 100, 30, key="semiurban"),
    "Village": st.slider("Village (%)", 0, 100, 20, key="village")
}

total = sum(route_types.values())
if total != 100:
    st.warning(f"⚠️ Total percentage must be 100%. Currently: {total}%")

# -------------------------------
# Save & Redirect
# -------------------------------
if st.button("Save", key="save_btn"):
    if total == 100:
        st.success("✅ Route details saved successfully.")

        st.session_state["box_data"] = box_data
        st.session_state["route_info"] = {
            "Source": source,
            "Destination": destination,
            "Route Distribution": route_types,
        }

        # ✅ Mark navigation flag
        st.session_state["go_save"] = True
        st.switch_page("pages\save.py")  # requires streamlit >= 1.32
    else:
        st.error("❌ Please adjust percentages to total 100%.")
