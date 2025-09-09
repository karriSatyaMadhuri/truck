import streamlit as st
import matplotlib
matplotlib.use("Agg")   # Needed for Streamlit
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="Truck Optimization & Box Placement", layout="wide")
st.title("📦 Truck Optimization & Box Placement")

# -------------------------------
# Get saved session data
# -------------------------------
box_data = st.session_state.get("box_data", [])
route_info = st.session_state.get("route_info", {})

if not box_data or not route_info:
    st.error("❌ No saved data found. Please go back and enter box & route details first.")
    st.stop()

# First box for optimization
box = box_data[0]

# -------------------------------
# Show box & route info
# -------------------------------
with st.container():
    st.subheader("📦 Selected Box & Route Info")
    st.write(f"*Box Type:* {box['type']}")
    st.write(f"*Quantity:* {box['quantity']}")
    st.write(f"*Dimensions:* {box['dimensions'][0]} × {box['dimensions'][1]} × {box['dimensions'][2]} mm")
    st.write(f"*Payload:* {box['payload']} kg")
    st.markdown("---")
    st.write(f"*Source:* {route_info['Source']}")
    st.write(f"*Destination:* {route_info['Destination']}")
    st.write("🛣 *Route Type Distribution:*")
    for route, pct in route_info["Route Distribution"].items():
        st.write(f"- {route}: {pct}%")

st.divider()

# -------------------------------
# Available Trucks
# -------------------------------
trucks = [
    {"name": "32 ft. Single Axle", "dimensions": (9750, 2440, 2440), "payload": 16000},
    {"name": "32 ft. Multi Axle", "dimensions": (9750, 2440, 2440), "payload": 21000},
    {"name": "22 ft. Truck", "dimensions": (7300, 2440, 2440), "payload": 10000},
]
st.subheader("🚛 Available Trucks")
cols = st.columns(len(trucks))
for col, truck in zip(cols, trucks):
    with col:
        st.markdown(f"### {truck['name']}")
        st.write(f"*Dimensions:* {truck['dimensions'][0]} × {truck['dimensions'][1]} × {truck['dimensions'][2]} mm")
        st.write(f"*Payload:* {truck['payload']} kg")

st.divider()

# -------------------------------
# Function to calculate arrangement
# -------------------------------
def calculate_arrangement(truck_dims, box_dims):
    truck_len, truck_wid, truck_hei = truck_dims
    box_len, box_wid, box_hei = box_dims

    fit_len = truck_len // box_len
    fit_wid = truck_wid // box_wid
    fit_hei = truck_hei // box_hei

    total_boxes = fit_len * fit_wid * fit_hei

    # Volumes in cubic meters
    truck_volume = (truck_len * truck_wid * truck_hei) / 1e9
    box_volume = (box_len * box_wid * box_hei) / 1e9
    used_volume = total_boxes * box_volume
    total_boxes
    utilisation_percent = (used_volume / truck_volume * 100) if truck_volume > 0 else 0

    return fit_len, fit_wid, fit_hei, total_boxes, truck_volume, box_volume, utilisation_percent

# -------------------------------
# Draw Rubik’s Cube Style
# -------------------------------
def draw_cube(fit_len, fit_wid, fit_hei):
    if fit_len == 0 or fit_wid == 0 or fit_hei == 0:
        st.error("❌ No boxes fit with these dimensions.")
        return

    voxels = np.ones((fit_len, fit_wid, fit_hei), dtype=bool)

    # SMALL cube size (4x4 inches)
    fig = plt.figure(figsize=(3, 3))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_box_aspect([fit_len, fit_wid, fit_hei])  # keeps proportions


    # Light green color (#90EE90)
    ax.voxels(voxels, facecolors="#90EE90", edgecolor="k")

    ax.set_xlabel(f"Length → ({fit_len} boxes)")
    ax.set_ylabel(f"Width → ({fit_wid} boxes)")
    ax.set_zlabel(f"Height → ({fit_hei} boxes)")

    ax.view_init(elev=25, azim=30)

    st.pyplot(fig)

# -------------------------------
# Streamlit App
# -------------------------------
def main():
    if st.button("🔍 Calculate & Visualize"):
        for truck in trucks:
            truck_len, truck_wid, truck_hei = truck["dimensions"]
            box_len, box_wid, box_hei = box["dimensions"]

            fit_len, fit_wid, fit_hei, total_boxes, truck_volume, box_volume, utilisation_percent = calculate_arrangement(
                (truck_len, truck_wid, truck_hei),
                (box_len, box_wid, box_hei)
            )

            if total_boxes > 0:
                st.success(f"✅ {truck['name']} → Total Boxes Fitted: {total_boxes}")

                # 📊 Key Info
                st.info(
                    f"**Truck Dimensions:** {truck_len} × {truck_wid} × {truck_hei} mm\n"
                    f"**Truck Volume:** {truck_volume:.2f} m³\n"
                    f"**Box Dimensions:** {box_len} × {box_wid} × {box_hei} mm\n"
                    f"**Box Volume:** {box_volume:.3f} m³\n"
                    f"**Utilisation:** {utilisation_percent:.1f}%\n"
                    f"👉 **Length-wise:** {fit_len} boxes\n"
                    f"👉 **Width-wise:** {fit_wid} boxes\n"
                    f"👉 **Height-wise:** {fit_hei} boxes\n"
                    f"👉 **Total Boxes in Truck:** {total_boxes}"
                    
                )

                st.subheader(f"📊 Arrangement in {truck['name']} ")
                draw_cube(fit_len, fit_wid, fit_hei)
            else:
                st.error(f"❌ No boxes can fit in {truck['name']}.")

if __name__ == "__main__":
    main()

