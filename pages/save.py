import streamlit as st
import itertools

# -------------------------------
# Page config
# -------------------------------
st.set_page_config(page_title="Truck Optimization & Box Placement", layout="wide")
st.title("📦 Truck Optimization & Box Placement")

colors = ["#90EE90", "#ADD8E6", "#FFA07A", "#FFD700", "#DDA0DD", "#FFB6C1"]

# -------------------------------
# Get saved session data
# -------------------------------
box_data = st.session_state.get("box_data", [])
route_info = st.session_state.get("route_info", {})

if not box_data or not route_info:
    st.error("❌ No saved data found. Please go back and enter box & route details first.")
    st.stop()

all_boxes = box_data

# -------------------------------
# Show box & route info
# -------------------------------
with st.container():
    st.subheader("📦 Selected Box & Route Info")
    for box in all_boxes:
        st.write(f"Box Type: {box['type']}")
        st.write(f"Quantity: {box['quantity']}")
        st.write(f"Dimensions: {box['dimensions'][0]} × {box['dimensions'][1]} × {box['dimensions'][2]} mm")
        st.write(f"Payload: {box['payload']} kg")
        st.markdown("---")

    st.write(f"Source: {route_info['Source']}")
    st.write(f"Destination: {route_info['Destination']}")
    st.write("🛣 Route Type Distribution:")
    for route, pct in route_info["Route Distribution"].items():
        st.write(f"- {route}: {pct}%")

st.divider()

# -------------------------------
# Define trucks
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
        st.write(f"Dimensions: {truck['dimensions'][0]} × {truck['dimensions'][1]} × {truck['dimensions'][2]} mm")
        st.write(f"Payload: {truck['payload']} kg")

st.divider()

# -------------------------------
# Payload toggle
# -------------------------------
apply_payload = st.checkbox("🚦 Apply Payload Restriction", value=True)

# -------------------------------
# Single Box Optimisation
# -------------------------------
def calculate_optimisation(truck, box, apply_payload=True):
    truck_len, truck_wid, truck_hei = truck["dimensions"]
    box_len, box_wid, box_hei = box["dimensions"]

    best_result = None

    # Try all 6 orientations
   # Use box dimensions as given (no orientation change)
    b_len, b_wid, b_hei =box_len, box_wid, box_hei

    fit_len = int(truck_len // b_len)
    fit_wid = int(truck_wid // b_wid)
    fit_hei = int(truck_hei // b_hei)
    boxes_by_space = fit_len * fit_wid * fit_hei

    if boxes_by_space <= 0:
        return None

    max_boxes_by_weight = int(truck["payload"] // float(box["payload"])) if box["payload"] > 0 else None
    if apply_payload and max_boxes_by_weight is not None:
        total_boxes = min(boxes_by_space, max_boxes_by_weight, box["quantity"])
    else:
        total_boxes = min(boxes_by_space, box["quantity"])

    truck_volume = (truck_len * truck_wid * truck_hei) / 1e9
    box_volume = (b_len * b_wid * b_hei) / 1e9
    utilised_volume = total_boxes * box_volume
    utilisation_percent = (utilised_volume / truck_volume) * 100 if truck_volume > 0 else 0

    result = {
            "truck_name": truck["name"],
            "truck_volume": round(truck_volume, 2),
            "box_volume": round(box_volume, 3),
            "boxes_by_space": boxes_by_space,
            "max_boxes_by_weight": max_boxes_by_weight,
            "boxes_per_truck": total_boxes,
            "utilisation_percent": round(utilisation_percent, 1),
            "orientation": (fit_len, fit_wid, fit_hei),
            "box_dims_used": (b_len,b_wid,b_hei)
        }
    return result 

    
def combined_optimisation(truck, all_boxes, apply_payload=True):
    truck_len, truck_wid, truck_hei = truck["dimensions"]
    payload_remaining = truck["payload"]

    # Strength ranking (strongest → weakest)
    strength_order = {"crate": 4, "pp": 3, "flc": 2, "plc": 1}

    # Sort by strength (desc), then volume
    sorted_boxes = sorted(
        all_boxes,
        key=lambda b: (-strength_order.get(b["type"].lower(), 0),
                       -(b['dimensions'][0] * b['dimensions'][1] * b['dimensions'][2]))
    )

    placed_counts = {}
    layers = []
    total_boxes = 0
    remaining_height = truck_hei

    # -----------------------------
    # STEP 1: Strongest box at bottom layer
    # -----------------------------
    if sorted_boxes:
        strongest_box = sorted_boxes[0]
        result = calculate_optimisation(truck, strongest_box, apply_payload)
        if result:
            b_len, b_wid, b_hei = result["box_dims_used"]

            # One full bottom layer of strongest box
            boxes_per_layer = (truck_len // b_len) * (truck_wid // b_wid)
            boxes_to_place = min(boxes_per_layer, strongest_box["quantity"])

            placed_counts[strongest_box["type"]] = boxes_to_place
            total_boxes += boxes_to_place
            payload_remaining -= boxes_to_place * strongest_box["payload"]

            # Visualisation for bottom layer
            layer = []
            placed = 0
            for row in range(int(truck_wid // b_wid)):
                grid_row = []
                for col in range(int(truck_len // b_len)):
                    if placed < boxes_to_place:
                        grid_row.append(strongest_box["type"])
                        placed += 1
                if grid_row:
                    layer.append(grid_row)
            if layer:
                layers.append((layer, strongest_box["type"]))

            # Reduce remaining height
            remaining_height -= b_hei

    # -----------------------------
    # STEP 2: Ensure all box types appear
    # -----------------------------
    for box in sorted_boxes[1:]:
        if remaining_height <= 0:
            break
        if payload_remaining <= 0:
            break

        result = calculate_optimisation(truck, box, apply_payload)
        if not result:
            continue

        b_len, b_wid, b_hei = result["box_dims_used"]
        if b_hei > remaining_height:
            continue

        # Place at least one row of this box
        boxes_per_layer = (truck_len // b_len) * (truck_wid // b_wid)
        boxes_to_place = min(boxes_per_layer, box["quantity"],
                             payload_remaining // max(1, box["payload"]))

        if boxes_to_place <= 0:
            continue

        placed_counts[box["type"]] = placed_counts.get(box["type"], 0) + boxes_to_place
        total_boxes += boxes_to_place
        payload_remaining -= boxes_to_place * box["payload"]

        # Visualisation for this layer
        layer = []
        placed = 0
        for row in range(int(truck_wid // b_wid)):
            grid_row = []
            for col in range(int(truck_len // b_len)):
                if placed < boxes_to_place:
                    grid_row.append(box["type"])
                    placed += 1
            if grid_row:
                layer.append(grid_row)
        if layer:
            layers.append((layer, box["type"]))

        remaining_height -= b_hei

    # -----------------------------
    # STEP 3: Fill remaining height greedily
    # -----------------------------
    while remaining_height > 0 and payload_remaining > 0:
        filled = False
        for box in sorted_boxes:
            result = calculate_optimisation(truck, box, apply_payload)
            if not result:
                continue
            b_len, b_wid, b_hei = result["box_dims_used"]
            if b_hei > remaining_height:
                continue

            boxes_per_layer = (truck_len // b_len) * (truck_wid // b_wid)
            boxes_to_place = min(boxes_per_layer, box["quantity"] - placed_counts.get(box["type"], 0),
                                 payload_remaining // max(1, box["payload"]))

            if boxes_to_place <= 0:
                continue

            placed_counts[box["type"]] = placed_counts.get(box["type"], 0) + boxes_to_place
            total_boxes += boxes_to_place
            payload_remaining -= boxes_to_place * box["payload"]

            # Visualisation
            layer = []
            placed = 0
            for row in range(int(truck_wid // b_wid)):
                grid_row = []
                for col in range(int(truck_len // b_len)):
                    if placed < boxes_to_place:
                        grid_row.append(box["type"])
                        placed += 1
                if grid_row:
                    layer.append(grid_row)
            if layer:
                layers.append((layer, box["type"]))

            remaining_height -= b_hei
            filled = True
            break  # move to next height after filling

        if not filled:  # nothing fits
            break

    return placed_counts, total_boxes, layers


# -------------------------------
# Optimisation Button
# -------------------------------
if st.button("🔵 Optimise Truck Loading", type="primary", use_container_width=True):
    st.subheader("📊 Truck Optimisation Results")

    for truck in trucks:
        st.markdown(f"## 🚛 {truck['name']}")

        # ---- LEFT (Specs + Per-Box Expanders) ----
        col_left, col_right = st.columns(2)

        with col_left:
            st.subheader("Truck Specs")
            st.info(
                f"Dimensions: {truck['dimensions'][0]} × {truck['dimensions'][1]} × {truck['dimensions'][2]} mm  \n"
                f"Payload: {truck['payload']} kg"
            )

            st.subheader("📦 Box Layout")
            for i, box in enumerate(all_boxes, start=1):
                result = calculate_optimisation(truck, box, apply_payload)
                if result:
                    with st.expander(f"📦 {box['type']} - {result['boxes_per_truck']} boxes in {truck['name']}"):
                        st.info(
                            f"Box {i} ({box['type']})  \n"
                            f"Dimensions: {result['box_dims_used']} mm  \n"
                            f"Quantity given: {box['quantity']}  \n"
                            f"Boxes along L × W × H: {result['orientation'][0]} × {result['orientation'][1]} × {result['orientation'][2]}  \n"
                            f"Boxes by Space: {result['boxes_by_space']}  \n"
                            f"Boxes by Payload: {result['max_boxes_by_weight']}  \n"
                            f"✅ Boxes Loaded: {result['boxes_per_truck']}  \n"
                            f"Utilisation: {result['utilisation_percent']} %"
                        )

        # ---- RIGHT (Combined Visualisation) ----
        with col_right:
            st.subheader("Combined Optimisation & Visualisation")

            placed_counts, total_boxes_loaded, layers = combined_optimisation(truck, all_boxes, apply_payload)

            st.success(f"✅ Total boxes placed inside truck: {total_boxes_loaded}")
            st.markdown("### Breakdown by Box Type")
            for btype, count in placed_counts.items():
                st.markdown(f"- {btype}: {count} boxes")

            color_map = {b['type']: colors[i % len(colors)] for i, b in enumerate(all_boxes)}

            for idx, (layer, box_type) in enumerate(layers, start=1):
                st.markdown(f"### Layer {idx} (Box type: {box_type}, Count: {sum(row.count(box_type) for row in layer)})")
                for row in layer:
                    cols = st.columns(len(row))
                    for col_idx, btype in enumerate(row):
                        with cols[col_idx]:
                            st.markdown(
                                f"<div style='background:{color_map[btype]}; border:1px solid #333; height:30px; "
                                "display:flex; justify-content:center; align-items:center;'>"
                                f"{btype[0]}</div>",
                                unsafe_allow_html=True
                            )
                st.markdown("---")

            st.markdown("### Legend")
            for box_type, colr in color_map.items():
                st.markdown(
                    f"<div style='display:inline-block; width:20px; height:20px; background:{colr}; "
                    "border:1px solid #333; margin-right:8px;'></div> "
                    f"{box_type}",
                    unsafe_allow_html=True
                )
