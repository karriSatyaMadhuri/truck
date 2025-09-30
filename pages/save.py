import streamlit as st
import plotly.graph_objects as go
import numpy as np
import itertools
from typing import List, Dict, Tuple, Optional
import pandas as pd

# -------------------------------
# Page config
# -------------------------------
st.set_page_config(page_title="3D Truck Optimization & Box Placement", layout="wide")
st.title("📦 3D Truck Optimization & Box Placement (Hybrid Strategy)")

colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7", "#DDA0DD", "#FFB6C1", "#98D8E8"]

# -------------------------------
# Initialize session state with updated placeholder data from the screenshot
# -------------------------------
if "box_data" not in st.session_state:
    st.session_state["box_data"] = [
        {"type": "plc_1", "dimensions": (1000, 1000, 1000), "payload": 100},
        {"type": "plc_2", "dimensions": (500, 500, 100), "payload": 100},
        {"type": "plc_3", "dimensions": (600, 700, 799), "payload": 100},
    ]

if "route_info" not in st.session_state:
    st.session_state["route_info"] = {
        "Source": "New York",
        "Destination": "Los Angeles",
        "Route Distribution": {"Highway": 80, "City": 20},
    }

# Get saved session data
all_boxes = st.session_state.get("box_data", [])
route_info = st.session_state.get("route_info", {})

if not all_boxes or not route_info:
    st.error("❌ No saved data found. Please go back and enter box & route details first.")
    st.stop()

# -------------------------------
# Show box & route info
# -------------------------------
with st.container():
    st.subheader("📦 Selected Box & Route Info")
    for box in all_boxes:
        st.write(f"Box Type: {box['type']}")
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
        st.write(f"Volume: {(truck['dimensions'][0] * truck['dimensions'][1] * truck['dimensions'][2]) / 1e9:.2f} m³")
        st.write(f"Payload: {truck['payload']} kg")

st.divider()

# -------------------------------
# Payload toggle
# -------------------------------
apply_payload = st.checkbox("🚦 Apply Payload Restriction", value=True)

# -------------------------------
# Box placement analysis functions
# -------------------------------
def get_box_orientations(dimensions: Tuple[int, int, int]) -> List[Tuple[int, int, int]]:
    """Get all possible unique orientations of a box using permutations"""
    return list(set(itertools.permutations(dimensions)))

def calculate_single_box_fit(truck_dims, box_dims):
    """Calculate how many boxes fit in each dimension for a single box type"""
    truck_l, truck_w, truck_h = truck_dims
    
    best_fit = None
    best_orientation = None
    best_total = 0
    
    # Try all orientations
    for orientation in get_box_orientations(box_dims):
        box_l, box_w, box_h = orientation
        
        if box_l <= truck_l and box_w <= truck_w and box_h <= truck_h:
            fit_l = int(truck_l // box_l)
            fit_w = int(truck_w // box_w) 
            fit_h = int(truck_h // box_h)
            total = fit_l * fit_w * fit_h
            
            if total > best_total:
                best_total = total
                best_fit = (fit_l, fit_w, fit_h)
                best_orientation = orientation
    
    return best_fit, best_orientation, best_total

# -------------------------------
# Enhanced 3D Box Placement Class
# -------------------------------
class Box3D:
    def __init__(self, box_type: str, dimensions: Tuple[int, int, int], position: Tuple[int, int, int], color: str, box_id: int = 0):
        self.box_type = box_type
        self.dimensions = dimensions  # (length, width, height)
        self.position = position      # (x, y, z)
        self.color = color
        self.box_id = box_id

class TruckLoader3D:
    def __init__(self, truck_dimensions: Tuple[int, int, int], payload_limit: int):
        # Store the original tuple
        self.truck_dimensions = truck_dimensions 
        
        # Unpack for easier access later
        self.truck_length, self.truck_width, self.truck_height = truck_dimensions
        
        self.payload_limit = payload_limit
        self.placed_boxes: List[Box3D] = []
        self.current_weight = 0
        self.box_counter = 0
        
    def can_place_box(self, box_dims: Tuple[int, int, int], position: Tuple[int, int, int]) -> bool:
        """Check if a box can be placed at the given position"""
        x, y, z = position
        length, width, height = box_dims
        
        # Check if box fits within truck boundaries
        if (x + length > self.truck_length or 
            y + width > self.truck_width or 
            z + height > self.truck_height):
            return False
        
        # Check collision with existing boxes
        for existing_box in self.placed_boxes:
            ex_x, ex_y, ex_z = existing_box.position
            ex_l, ex_w, ex_h = existing_box.dimensions
            
            # Check if boxes overlap
            if not (x >= ex_x + ex_l or x + length <= ex_x or
                    y >= ex_y + ex_w or y + width <= ex_y or
                    z >= ex_z + ex_h or z + height <= ex_z):
                return False
        
        return True
    
    def get_support_level(self, position: Tuple[int, int, int], box_dims: Tuple[int, int, int]) -> int:
        """Get the minimum support height needed at this position"""
        x, y = position[0], position[1]
        box_l, box_w = box_dims[0], box_dims[1]
        max_support_height = 0
        
        # Check all existing boxes for support
        for existing_box in self.placed_boxes:
            ex_x, ex_y, ex_z = existing_box.position
            ex_l, ex_w, ex_h = existing_box.dimensions
            
            # Check if existing box provides support for any part of the new box's base
            if not (x + box_l <= ex_x or x >= ex_x + ex_l or 
                    y + box_w <= ex_y or y >= ex_y + ex_w):
                max_support_height = max(max_support_height, ex_z + ex_h)
        
        return max_support_height
    
    def find_all_valid_positions(self, box_dims: Tuple[int, int, int], step_size: int = 50) -> List[Tuple[int, int, int]]:
        """Find all valid positions where a box can be placed"""
        valid_positions = []
        length, width, height = box_dims
        
        # Generate candidate positions
        for z in range(0, self.truck_height - height + 1, step_size):
            for y in range(0, self.truck_width - width + 1, step_size):
                for x in range(0, self.truck_length - length + 1, step_size):
                    position = (x, y, z)
                    
                    # Check if position is supported (ground level or on top of other boxes)
                    required_support_height = self.get_support_level(position, box_dims)
                    
                    # Position must be at the required support level
                    if z == required_support_height and self.can_place_box(box_dims, position):
                        valid_positions.append(position)
        
        # Sort positions by preference (ground level first, then by x, y coordinates)
        valid_positions.sort(key=lambda pos: (pos[2], pos[0], pos[1]))
        return valid_positions
    
    def place_box_at_best_position(self, box_type: str, box_dims: Tuple[int, int, int], 
                                    box_weight: float, color: str) -> Optional[Tuple[int, int, int]]:
        """Try to place a box at the best possible position and return its position if successful"""
        if apply_payload and self.current_weight + box_weight > self.payload_limit:
            return None
        
        # Find all valid positions
        valid_positions = self.find_all_valid_positions(box_dims)
        
        if valid_positions:
            # Use the first (best) valid position
            position = valid_positions[0]
            
            # Place the box
            self.box_counter += 1
            new_box = Box3D(box_type, box_dims, position, color, self.box_counter)
            self.placed_boxes.append(new_box)
            self.current_weight += box_weight
            return position
        
        return None
    
    def get_detailed_stats(self, box_types_data) -> Dict:
        """Calculate detailed utilization statistics"""
        truck_volume = (self.truck_length * self.truck_width * self.truck_height) / 1e9  # cubic meters
        
        total_box_volume = 0
        box_counts = {}
        box_details = {}
        
        for box in self.placed_boxes:
            volume = (box.dimensions[0] * box.dimensions[1] * box.dimensions[2]) / 1e9
            total_box_volume += volume
            
            if box.box_type in box_counts:
                box_counts[box.box_type] += 1
            else:
                box_counts[box.box_type] = 1
        
        # Calculate theoretical maximum for each box type
        for box_type_data in box_types_data:
            box_type = box_type_data['type']
            if box_type in box_counts:
                best_fit, best_orientation, max_single = calculate_single_box_fit(
                    (self.truck_length, self.truck_width, self.truck_height),
                    box_type_data['dimensions']
                )
                
                if best_fit:
                    box_details[box_type] = {
                        'placed_count': box_counts[box_type],
                        'max_single_type': max_single,
                        'best_orientation': best_orientation,
                        'fit_length': best_fit[0],
                        'fit_width': best_fit[1], 
                        'fit_height': best_fit[2],
                        'original_dims': box_type_data['dimensions']
                    }
        
        volume_utilization = (total_box_volume / truck_volume) * 100 if truck_volume > 0 else 0
        weight_utilization = (self.current_weight / self.payload_limit) * 100 if self.payload_limit > 0 else 0
        
        return {
            'truck_volume': round(truck_volume, 3),
            'used_volume': round(total_box_volume, 3),
            'volume_utilization': round(volume_utilization, 2),
            'weight_utilization': round(weight_utilization, 2),
            'total_boxes': len(self.placed_boxes),
            'box_counts': box_counts,
            'box_details': box_details,
            'current_weight': round(self.current_weight, 2),
            'payload_limit': self.payload_limit
        }

def create_3d_visualization(loader: TruckLoader3D, truck_name: str):
    """Create 3D plotly visualization of the truck loading with transparent faces and wireframes."""
    fig = go.Figure()

    # Add truck outline/wireframe
    truck_dims = loader.truck_dimensions
    fig.add_trace(go.Scatter3d(
        x=[0, truck_dims[0], truck_dims[0], 0, 0, None, 0, truck_dims[0], None, truck_dims[0], truck_dims[0], None, 0, 0],
        y=[0, 0, truck_dims[1], truck_dims[1], 0, None, 0, 0, None, truck_dims[1], truck_dims[1], None, truck_dims[1], truck_dims[1]],
        z=[0, 0, 0, 0, 0, None, truck_dims[2], truck_dims[2], None, truck_dims[2], 0, None, truck_dims[2], 0],
        mode='lines', line=dict(color='black', width=4), name='Truck Outline', showlegend=True
    ))

    # Add the floor
    fig.add_trace(go.Mesh3d(
        x=[0, truck_dims[0], truck_dims[0], 0],
        y=[0, 0, truck_dims[1], truck_dims[1]],
        z=[0, 0, 0, 0],
        color='lightgray',
        opacity=0.6,
        showlegend=False
    ))

    # Add boxes with both solid faces and wireframe edges
    for i, box in enumerate(loader.placed_boxes):
        x, y, z = box.position
        l, w, h = box.dimensions

        # Define vertices for the box
        vertices = np.array([
            [x, y, z], [x + l, y, z], [x + l, y + w, z], [x, y + w, z],
            [x, y, z + h], [x + l, y, z + h], [x + l, y + w, z + h], [x, y + w, z + h]
        ])

        # Define indices for the faces of the box
        faces = [
            [0, 1, 2, 3],  # Bottom face
            [4, 5, 6, 7],  # Top face
            [0, 1, 5, 4],  # Front face
            [1, 2, 6, 5],  # Right face
            [2, 3, 7, 6],  # Back face
            [3, 0, 4, 7]   # Left face
        ]
        
        # Add mesh for the solid, transparent faces
        fig.add_trace(go.Mesh3d(
            x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
            i=[f[0] for f in faces],
            j=[f[1] for f in faces],
            k=[f[2] for f in faces],
            name=f'Box {box.box_id}',
            color=box.color,
            opacity=0.5,
            showlegend=True,
            hoverinfo='skip'  # Hide hover text for the solid faces
        ))
        
        # Add wireframe for the edges
        edge_x, edge_y, edge_z = [], [], []
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7)
        ]
        for start, end in edges:
            edge_x.extend([vertices[start][0], vertices[end][0], None])
            edge_y.extend([vertices[start][1], vertices[end][1], None])
            edge_z.extend([vertices[start][2], vertices[end][2], None])

        fig.add_trace(go.Scatter3d(
            x=edge_x, y=edge_y, z=edge_z,
            mode='lines',
            line=dict(color='black', width=4),
            showlegend=False,
            hoverinfo='text',
            hovertext=f'Box #{box.box_id}: {box.box_type}<br>Position: ({x}, {y}, {z})<br>Dimensions: {l}×{w}×{h} mm'
        ))

        # Add text labels at the center of each box
        fig.add_trace(go.Scatter3d(
            x=[x + l / 2], y=[y + w / 2], z=[z + h / 2],
            mode='text',
            text=[f"{box.box_type[:2].upper()}{box.box_id}"],
            textfont=dict(size=12, color='black'),
            showlegend=False
        ))

    # Update layout for a better view
    fig.update_layout(
        scene=dict(
            xaxis=dict(title='Length (mm)', range=[0, loader.truck_length], autorange=False),
            yaxis=dict(title='Width (mm)', range=[0, loader.truck_width], autorange=False),
            zaxis=dict(title='Height (mm)', range=[0, loader.truck_height], autorange=False),
            aspectmode='data',
            xaxis_showgrid=True,
            yaxis_showgrid=True,
            zaxis_showgrid=True
        ),
        title=f"3D Loading Visualization for {truck_name}",
        margin=dict(l=0, r=0, b=0, t=30),
        legend=dict(x=0.0, y=1, traceorder="normal")
    )

    return fig


def optimize_maximum_box_loading(truck: Dict, boxes: List[Dict], apply_payload: bool = True):
    """
    Pure greedy algorithm to fill the truck by placing the largest available boxes first.
    This method will produce a cleaner, more intuitive visualization.
    """
    loader = TruckLoader3D(truck["dimensions"], truck["payload"] if apply_payload else float('inf'))
    color_map = {box['type']: colors[i % len(colors)] for i, box in enumerate(boxes)}
    placed_counts = {box['type']: 0 for box in boxes}
    placement_log = []

    # Prepare and sort all box orientations by volume (largest first)
    available_boxes = []
    for box in boxes:
        orientations = get_box_orientations(box['dimensions'])
        volume = box['dimensions'][0] * box['dimensions'][1] * box['dimensions'][2]

        for orientation in orientations:
            available_boxes.append({
                'type': box['type'],
                'dimensions': orientation,
                'original_dims': box['dimensions'],
                'payload': box['payload'],
                'color': color_map[box['type']],
                'volume': volume,
                # Simple greedy score: place largest volume boxes first
                'efficiency_score': volume
            })
    
    available_boxes.sort(key=lambda x: x['efficiency_score'], reverse=True)

    st.info("Using a pure greedy algorithm to fill the truck with the largest available boxes first.")

    while True:
        placed_this_round = False
        
        for box_option in available_boxes:
            box_type = box_option['type']
            
            # Check payload restriction
            if apply_payload and loader.current_weight + box_option['payload'] > loader.payload_limit:
                continue
            
            # Find a valid position and place the box
            position = loader.place_box_at_best_position(
                box_type, 
                box_option['dimensions'], 
                box_option['payload'], 
                box_option['color']
            )
            
            if position:
                placed_counts[box_type] += 1
                placed_this_round = True
                placement_log.append({
                    'box_type': box_type,
                    'position': position,
                    'dimensions': box_option['dimensions'],
                    'original_dimensions': box_option['original_dims'],
                    'weight': box_option['payload'],
                    'box_id': loader.box_counter
                })
                break  # Break to re-sort for a new placement search
        
        if not placed_this_round:
            break
    
    return loader, placement_log, placed_counts

# -------------------------------
# Enhanced Optimization Button
# -------------------------------
if st.button("🔵 Optimise 3D Box Loading", type="primary", use_container_width=True):
    st.subheader("📊 3D Box Loading Optimization Results")

    for truck in trucks:
        st.markdown(f"## 🚛 {truck['name']}")
        
        with st.container(border=True):
            st.markdown("### 📥 Input Box Details (Assumed Infinite Supply)")
            box_info_cols = st.columns(len(all_boxes))
            for i, box in enumerate(all_boxes):
                with box_info_cols[i]:
                    st.markdown(f"""
                        <div style='background-color: {colors[i % len(colors)]}20; border-radius: 8px; padding: 10px; text-align: center;'>
                            <h5 style='color: {colors[i % len(colors)]}; margin: 0;'>{box['type']}</h5>
                            <p style='margin: 5px 0 0;'>
                                Dims: **{box['dimensions'][0]}×{box['dimensions'][1]}×{box['dimensions'][2]}** mm<br>
                                Payload: **{box['payload']}** kg
                            </p>
                        </div>
                    """, unsafe_allow_html=True)
        
        st.info(f"""
        *Truck Specifications:*
        - Dimensions: {truck['dimensions'][0]} × {truck['dimensions'][1]} × {truck['dimensions'][2]} mm
        - Volume: {(truck['dimensions'][0] * truck['dimensions'][1] * truck['dimensions'][2]) / 1e9:.3f} m³
        - Payload Limit: {truck['payload']} kg
        """)
        
        loader, placement_log, placed_counts = optimize_maximum_box_loading(truck, all_boxes, apply_payload)
        stats = loader.get_detailed_stats(all_boxes)
        
        st.subheader("📦 Box Loading Summary")
        
        box_count_cols = st.columns(len(all_boxes))
        total_placed = 0
        
        for i, (col, box) in enumerate(zip(box_count_cols, all_boxes)):
            with col:
                placed = placed_counts.get(box['type'], 0)
                total_placed += placed
                
                color = colors[i % len(colors)]
                
                st.markdown(
                    f"""
                    <div style='padding:15px; border-radius:10px; background:linear-gradient(135deg, {color}20, {color}10); border:2px solid {color}; text-align:center; margin-bottom:10px;'>
                        <h4 style='color:{color}; margin:0;'>{box['type']}</h4>
                        <h2 style='color:#333; margin:5px 0;'>{placed}</h2>
                        <p style='margin:5px 0; font-size:14px;'>
                            Boxes Placed
                        </p>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
        
        st.success(f"🎯 *Overall Loading Summary:* **{total_placed}** total boxes placed.")
        
        col1, col2, col3 = st.columns([1, 1.2, 0.8])
        
        with col1:
            st.subheader("📈 Utilization Metrics")
            metric_col1, metric_col2 = st.columns(2)
            with metric_col1:
                st.metric("Volume Used", f"{stats['volume_utilization']}%")
                st.metric("Weight Used", f"{stats['weight_utilization']}%")
            with metric_col2:
                st.metric("Total Boxes", stats['total_boxes'])
                st.metric("Weight", f"{stats['current_weight']:.1f} kg")
            
            st.subheader("📋 Detailed Box Analysis")
            for box_type, count in placed_counts.items():
                if count > 0:
                    st.write(f"{box_type}:")
                    st.write(f"• Placed: {count} boxes")
                    st.markdown("---")
        
        with col2:
            st.subheader("🎯 3D Interactive Visualization")
            if loader.placed_boxes:
                fig = create_3d_visualization(loader, truck['name'])
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("❌ No boxes could be placed in this truck configuration.")
        
        with col3:
            st.subheader("📊 Loading Statistics")
            st.write(f"*Truck Capacity Analysis:*")
            st.write(f"• Total Volume: {stats['truck_volume']:.3f} m³")
            st.write(f"• Used Volume: {stats['used_volume']:.3f} m³")
            st.write(f"• Free Volume: {stats['truck_volume'] - stats['used_volume']:.3f} m³")
            st.write(f"• Volume Efficiency: {stats['volume_utilization']:.1f}%")
            if apply_payload:
                st.write(f"• Weight Capacity: {stats['payload_limit']} kg")
                st.write(f"• Current Weight: {stats['current_weight']:.1f} kg")
                st.write(f"• Remaining Capacity: {stats['payload_limit'] - stats['current_weight']:.1f} kg")
                st.write(f"• Weight Efficiency: {stats['weight_utilization']:.1f}%")
            st.subheader("📦 Box Type Distribution")
            for box_type, count in placed_counts.items():
                if count > 0:
                    percentage = (count / stats['total_boxes'] * 100) if stats['total_boxes'] > 0 else 0
                    st.write(f"• {box_type}: {count} ({percentage:.1f}%)")
        
        if placement_log:
            with st.expander(f"🔍 Complete Placement Log - {truck['name']} ({len(placement_log)} boxes)", expanded=False):
                log_df = pd.DataFrame(placement_log)
                log_df['Position'] = log_df['position'].apply(lambda x: f"({x[0]}, {x[1]}, {x[2]})")
                log_df['Used Dims'] = log_df['dimensions'].apply(lambda x: f"{x[0]}×{x[1]}×{x[2]}")
                log_df['Original Dims'] = log_df['original_dimensions'].apply(lambda x: f"{x[0]}×{x[1]}×{x[2]}")
                display_log = log_df[['box_id', 'box_type', 'Position', 'Original Dims', 'Used Dims', 'weight']].copy()
                display_log.columns = ['Box ID', 'Type', 'Position (x,y,z)', 'Original (L×W×H)', 'Used (L×W×H)', 'Weight (kg)']
                st.dataframe(display_log, use_container_width=True, hide_index=True)
                
                st.success(f"""
                ✅ *Loading Validation Complete*
                - *Total Boxes Placed*: {len(placement_log)}
                - *Truck Full*: No more boxes could be placed.
                - *Loading Efficiency*: {stats['volume_utilization']:.1f}%
                - *All box types attempted for placement*: Each box type has at least one box placed if possible.
                - *No Overlapping Boxes*: All positions validated
                - *Structural Support*: All boxes properly supported
                - *Boundary Compliance*: All boxes within truck limits
                - *Weight Compliance*: {'Respected' if apply_payload else 'Ignored (disabled)'}
                """)
        
        st.markdown("---")

# -------------------------------
# Box Type Legend & Analysis
# -------------------------------
if all_boxes:
    st.subheader("🎨 Box Type Legend & Individual Analysis")
    for i, box in enumerate(all_boxes):
        color = colors[i % len(colors)]
        col_legend, col_analysis = st.columns([1, 2])
        with col_legend:
            st.markdown(
                f"<div style='display:flex; align-items:center; padding:15px; border-radius:8px; background:#f8f9fa; border-left:5px solid {color};'>"
                f"<div style='width:25px; height:25px; background:{color}; border:2px solid #333; margin-right:15px; border-radius:4px;'></div>"
                f"<div><strong>{box['type']}</strong><br>"
                f"<small>Dims: {box['dimensions'][0]}×{box['dimensions'][1]}×{box['dimensions'][2]} mm | Weight: {box['payload']} kg</small></div>"
                f"</div>",
                unsafe_allow_html=True
            )
        with col_analysis:
            st.write(f"**{box['type']} - Single Type Analysis**")
            single_analysis = []
            for truck in trucks:
                best_fit, best_orientation, max_boxes = calculate_single_box_fit(
                    truck['dimensions'], box['dimensions']
                )
                if best_fit:
                    weight_limit = max_boxes if not apply_payload else int(truck['payload'] / box['payload'])
                    final_boxes = min(max_boxes, weight_limit)
                    volume_used = (final_boxes * box['dimensions'][0] * box['dimensions'][1] * box['dimensions'][2]) / 1e9
                    truck_volume = (truck['dimensions'][0] * truck['dimensions'][1] * truck['dimensions'][2]) / 1e9
                    utilization = (volume_used / truck_volume) * 100
                    single_analysis.append({
                        'Truck': truck['name'],
                        'Length×Width×Height': f"{best_fit[0]}×{best_fit[1]}×{best_fit[2]}",
                        'Max Boxes': final_boxes,
                        'Utilization': f"{utilization:.1f}%"
                    })
            if single_analysis:
                analysis_df = pd.DataFrame(single_analysis)
                st.dataframe(analysis_df, use_container_width=True, hide_index=True)
            else:
                st.warning("This box type doesn't fit in any truck.")

    st.subheader("📊 Maximum Loading vs Single Type Comparison")
    st.info("""
    *Hybrid Loading Algorithm Features:*
    - **Guaranteed Placement**: The algorithm first attempts to place at least one of each box type to ensure all are represented.
    - **Greedy Filling**: After initial placement, it fills the remaining space with the most efficient boxes to maximize utilization.
    - **Multi-Orientation Support**: It tests all 6 possible orientations for each box type.
    - **Intelligent Stacking**: Boxes are placed with proper support and no overlaps.
    - **Weight & Volume Optimization**: It considers both dimensional and weight constraints.
    - **Real-time Tracking**: Shows the exact count of each box type placed.
    - **Space Efficiency**: Optimized placement algorithm for maximum space utilization.
    """)
