import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib.animation import FuncAnimation
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import io
import os
from PIL import Image
import random

# Set page configuration
st.set_page_config(
    page_title="Vehicle Routing Optimizer",
    page_icon="🚚",
    layout="wide"
)

# Add custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #42A5F5;
    }
    .info-box {
        background-color: #E3F2FD;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .success-box {
        background-color: #E8F5E9;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .warning-box {
        background-color: #FFF8E1;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .metrics-container {
        display: flex;
        justify-content: space-between;
    }
    .metric-card {
        background-color: #fff;
        border-radius: 0.5rem;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        padding: 1rem;
        width: 30%;
        text-align: center;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #1E88E5;
    }
    .metric-label {
        font-size: 1rem;
        color: #616161;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("<h1 class='main-header'>🚚 Vehicle Routing Optimizer</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Optimize delivery routes using Matrix-Based Ant Colony Optimization</p>", unsafe_allow_html=True)

# Create tabs
tabs = st.tabs(["📊 Data Upload", "⚙️ Configure & Run", "🚗 Results & Visualization"])

# Enhanced Order Preprocessing
def preprocess_orders(orders_df):
    """
    Preprocess order data to ensure quality and consistency
    
    Args:
        orders_df: DataFrame with order information
        
    Returns:
        Processed DataFrame
    """
    # Create a copy to avoid modifying the original
    df = orders_df.copy()
    
    # Remove duplicate orders
    df = df.drop_duplicates(subset=['Order_ID'])
    
    # Ensure weight values are reasonable (convert to kg if needed)
    if 'Weight' in df.columns:
        # Convert very small weights that might be in tons to kg
        if df['Weight'].median() < 100:
            df['Weight'] = df['Weight'] * 1000
            
        # Cap unreasonably large weights
        max_reasonable_weight = 30000  # 30 tons in kg
        df.loc[df['Weight'] > max_reasonable_weight, 'Weight'] = max_reasonable_weight
    
    # Ensure required columns exist
    required_cols = ['Order_ID', 'Source', 'Destination']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column {col} missing from orders data")
    
    # Validate source and destination are not the same
    same_locations = df['Source'] == df['Destination']
    if same_locations.any():
        print(f"Warning: {same_locations.sum()} orders have the same source and destination")
    
    return df

# City Deduplication for Route Generation
def deduplicate_route_cities(route_cities):
    """
    Ensure each destination appears only once in a route sequence
    
    Args:
        route_cities: List of cities in route
        
    Returns:
        Deduplicated route sequence
    """
    # First city is warehouse/depot - keep it
    start_city = route_cities[0]
    
    # Last city should return to warehouse
    end_city = route_cities[-1]
    
    # Middle cities should have no duplicates
    middle_cities = route_cities[1:-1]
    unique_middle_cities = []
    
    # Track cities we've already seen
    seen_cities = set()
    for city in middle_cities:
        if city not in seen_cities or city == start_city:
            unique_middle_cities.append(city)
            seen_cities.add(city)
    
    # Reconstruct route
    return [start_city] + unique_middle_cities + [end_city]

# Read distance.csv and create distance matrix
def load_distance_matrix(df):
    """
    Create distance matrix from distance dataframe
    
    Args:
        df: DataFrame with Source, Destination, Distance columns
        
    Returns:
        dist_matrix: numpy array distance matrix
        city_to_index: dictionary mapping city names to matrix indices
        cities: list of city names
    """
    cities = pd.unique(df[['Source', 'Destination']].values.ravel('K'))
    city_to_index = {city: idx for idx, city in enumerate(cities)}
    n_cities = len(cities)
    
    # Create distance matrix
    dist_matrix = np.ones((n_cities, n_cities)) * 1e6  # Initialize with large values
    np.fill_diagonal(dist_matrix, 0)  # Zero distance to self
    
    # Fill matrix with distances
    for _, row in df.iterrows():
        i = city_to_index[row['Source']]
        j = city_to_index[row['Destination']]
        distance = row['Distance(M)']  # Match the column name in your data
        dist_matrix[i][j] = distance
        dist_matrix[j][i] = distance  # Assuming symmetrical distances
    
    return dist_matrix, city_to_index, cities

# Read orders CSV and create customer list
def load_orders(df, city_to_index):
    """
    Create customer list from orders dataframe
    
    Args:
        df: DataFrame with order information
        city_to_index: dictionary mapping city names to matrix indices
        
    Returns:
        customers: list of customer dictionaries
    """
    customers = []
    for idx, row in df.iterrows():
        # Skip if destination not in city_to_index
        if row['Destination'] not in city_to_index:
            continue
            
        customers.append({
            'order_id': row['Order_ID'],
            'source': row['Source'],
            'destination': row['Destination'],
            'demand': row['Weight'] / 1000,  # Convert to more manageable units
            'city_index': city_to_index[row['Destination']]
        })
    return customers

# Function to generate positions for cities
def generate_city_positions(cities, customers):
    """
    Generate visual positions for cities
    
    Args:
        cities: list of city names
        customers: list of customer dictionaries
        
    Returns:
        positions: dictionary mapping city names to (x, y) coordinates
    """
    # Identify warehouse cities (sources)
    warehouse_cities = set()
    for customer in customers:
        warehouse_cities.add(customer['source'])
    
    # Count customers per city for sizing
    city_counts = {}
    for customer in customers:
        dest = customer['destination']
        if dest not in city_counts:
            city_counts[dest] = 0
        city_counts[dest] += 1
    
    # Generate positions
    positions = {}
    
    # Position warehouses in a circle
    n_warehouses = len(warehouse_cities)
    if n_warehouses > 0:
        radius = 0.4
        for i, city in enumerate(warehouse_cities):
            angle = 2 * np.pi * i / n_warehouses
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            positions[city] = (x, y)
    
    # Position other cities in a larger circle/grid
    other_cities = [city for city in cities if city not in positions]
    grid_size = int(np.ceil(np.sqrt(len(other_cities))))
    grid_step = 1.0 / grid_size if grid_size > 0 else 0.1
    
    i = 0
    for city in other_cities:
        row = i // grid_size
        col = i % grid_size
        x = (col - grid_size/2) * grid_step * 2
        y = (row - grid_size/2) * grid_step * 2
        
        # Scale position by city demand if available
        scale = np.log1p(city_counts.get(city, 1)) / 10 if city in city_counts else 0
        x += np.random.uniform(-0.02, 0.02) * (1 + scale)
        y += np.random.uniform(-0.02, 0.02) * (1 + scale)
        
        positions[city] = (x, y)
        i += 1
    
    return positions

# Advanced ACO solver class
class EnhancedACOSolver:
    """
    Enhanced Ant Colony Optimization solver for Vehicle Routing Problems
    """
    
    def __init__(self, dist_matrix, customers, city_to_index, cities, 
                 num_trucks=5, truck_capacity=15000, num_ants=20, 
                 num_iterations=50, alpha=1.0, beta=3.0, evaporation=0.5, 
                 pheromone_init=1.0, progress_bar=None):
        """
        Initialize ACO solver with improved default parameters
        
        Args:
            dist_matrix: distance matrix
            customers: list of customer dictionaries
            city_to_index: dictionary mapping city names to matrix indices
            cities: list of city names
            num_trucks: number of trucks
            truck_capacity: capacity of each truck
            num_ants: number of ants
            num_iterations: number of iterations
            alpha: pheromone importance factor
            beta: heuristic importance factor (increased for better distance focus)
            evaporation: pheromone evaporation rate
            pheromone_init: initial pheromone value
            progress_bar: streamlit progress bar
        """
        self.dist_matrix = dist_matrix
        self.customers = customers
        self.city_to_index = city_to_index
        self.cities = cities
        self.num_trucks = num_trucks
        self.truck_capacity = truck_capacity
        self.num_ants = num_ants
        self.num_iterations = num_iterations
        self.alpha = alpha
        self.beta = beta  # Increased to prioritize shorter routes
        self.evaporation = evaporation
        self.pheromone_init = pheromone_init
        self.progress_bar = progress_bar
        
        # Identify warehouse cities (sources)
        self.warehouse_indices = set()
        for customer in customers:
            if 'source' in customer and customer['source'] in city_to_index:
                self.warehouse_indices.add(city_to_index[customer['source']])
        
        # If no warehouse indices, assume node 0 is depot
        if not self.warehouse_indices:
            self.warehouse_indices = {0}
        
        # Initialize pheromone matrix with floating-point dtype
        n_nodes = len(dist_matrix)
        self.pheromones = np.full((n_nodes, n_nodes), pheromone_init, dtype=float)
        
        # Best solution tracking
        self.best_solution = None
        self.best_distance = float('inf')
        self.solution_history = []

    def solve(self):
        """
        Run ACO algorithm to solve the Vehicle Routing Problem
        
        Returns:
            Dictionary with solution details
        """
        start_time = time.time()
        
        for iteration in range(self.num_iterations):
            ant_solutions = []
            
            for ant in range(self.num_ants):
                # Construct solution for this ant
                solution = self._construct_solution(list(range(len(self.customers))))
                
                # Calculate solution cost
                distance = self._calculate_solution_cost(solution)
                
                ant_solutions.append((solution, distance))
                
                # Update best solution if better
                if distance < self.best_distance:
                    self.best_distance = distance
                    self.best_solution = solution
            
            # Update pheromones
            self._update_pheromones(ant_solutions)
            
            # Track solution history
            self.solution_history.append((self.best_distance, iteration))
            
            # Update progress bar
            if self.progress_bar:
                self.progress_bar.progress((iteration + 1) / self.num_iterations)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Prepare detailed route information
        route_details = self._prepare_route_details()
        
        # Get unassigned orders
        unassigned = list(self.best_solution['unassigned_customers'])
        
        return {
            'best_solution': self.best_solution,
            'best_distance': self.best_distance,
            'solution_history': self.solution_history,
            'unassigned': unassigned,
            'route_details': route_details,
            'execution_time': execution_time
        }

    def _construct_solution(self, remaining_customers):
        """
        Construct a VRP solution using ACO with improved route logic
        
        Args:
            remaining_customers: List of customers to assign
            
        Returns:
            Dictionary with solution structure
        """
        # Initialize solution structure
        solution = {
            'routes': [],  # List of routes
            'truck_loads': [],  # List of truck loads
            'assigned_customers': set(),  # Set of assigned customer indices
            'unassigned_customers': set(remaining_customers)  # Set of unassigned customer indices
        }
        
        # Sort customers by demand (descending) for better bin packing
        sorted_customers = sorted(
            remaining_customers,
            key=lambda i: self.customers[i]['demand'],
            reverse=True
        )
        
        # Create routes for each truck
        for truck_idx in range(self.num_trucks):
            # Select a random warehouse as starting point
            warehouse_idx = random.choice(list(self.warehouse_indices))
            
            # Initialize route with starting warehouse
            route = [warehouse_idx]
            current_location = warehouse_idx
            route_load = 0
            route_customers = set()
            
            # Track visited cities to avoid duplicates
            visited_cities = {warehouse_idx}
            
            # Continue adding customers until no more fit or none available
            while sorted_customers:
                best_customer_idx = None
                best_score = float('inf')
                
                # Evaluate each remaining customer
                for i, customer_idx in enumerate(sorted_customers):
                    customer = self.customers[customer_idx]
                    destination = customer['city_index']
                    
                    # Skip if adding this customer exceeds capacity
                    if route_load + customer['demand'] > self.truck_capacity:
                        continue
                    
                    # Skip if we've already visited this city in this route
                    if destination in visited_cities:
                        continue
                    
                    # Calculate score based on distance and pheromones
                    distance = self.dist_matrix[current_location][destination]
                    if distance <= 0:  # Avoid division by zero
                        distance = 0.1
                    
                    pheromone = self.pheromones[current_location][destination]
                    
                    # ACO formula: balance pheromone and distance
                    score = distance / ((pheromone ** self.alpha) * ((1/distance) ** self.beta))
                    
                    if score < best_score:
                        best_score = score
                        best_customer_idx = customer_idx
                
                # If no suitable customer found, break the loop
                if best_customer_idx is None:
                    break
                
                # Add best customer to route
                best_customer = self.customers[best_customer_idx]
                destination = best_customer['city_index']
                
                route.append(destination)
                visited_cities.add(destination)
                current_location = destination
                route_load += best_customer['demand']
                
                # Track assigned customers
                route_customers.add(best_customer_idx)
                sorted_customers.remove(best_customer_idx)
                solution['assigned_customers'].add(best_customer_idx)
                solution['unassigned_customers'].remove(best_customer_idx)
                
                # Limit route length for practical operations
                if len(route) > 15:
                    break
            
            # Return to warehouse
            route.append(warehouse_idx)
            
            # Only add route if it has customers
            if len(route) > 2:
                solution['routes'].append(route)
                solution['truck_loads'].append(route_load)
        
        # Try to assign remaining customers with insertion heuristic
        if solution['unassigned_customers'] and solution['routes']:
            self._insert_remaining_customers(solution)
        
        return solution

    def _insert_remaining_customers(self, solution):
        """
        Try to insert remaining customers into existing routes
        
        Args:
            solution: Current solution dictionary
        """
        for customer_idx in list(solution['unassigned_customers']):
            customer = self.customers[customer_idx]
            
            # Find best insertion point across all routes
            best_route_idx = None
            best_position = None
            best_increase = float('inf')
            
            for route_idx, route in enumerate(solution['routes']):
                # Skip if adding this customer exceeds route capacity
                if solution['truck_loads'][route_idx] + customer['demand'] > self.truck_capacity:
                    continue
                
                # Try to insert at each position
                for pos in range(1, len(route)):
                    prev_city = route[pos-1]
                    next_city = route[pos]
                    city_idx = customer['city_index']
                    
                    # Skip if city already in route
                    if city_idx in route:
                        continue
                    
                    # Calculate insertion cost
                    original_dist = self.dist_matrix[prev_city][next_city]
                    new_dist = self.dist_matrix[prev_city][city_idx] + self.dist_matrix[city_idx][next_city]
                    increase = new_dist - original_dist
                    
                    if increase < best_increase:
                        best_increase = increase
                        best_route_idx = route_idx
                        best_position = pos
            
            # Insert customer if good position found
            if best_route_idx is not None:
                solution['routes'][best_route_idx].insert(best_position, customer['city_index'])
                solution['truck_loads'][best_route_idx] += customer['demand']
                solution['assigned_customers'].add(customer_idx)
                solution['unassigned_customers'].remove(customer_idx)

    def _calculate_solution_cost(self, solution):
        """
        Calculate total cost of a solution
        
        Args:
            solution: Solution dictionary
            
        Returns:
            Total cost value
        """
        total_distance = 0
        
        # Sum distance for each route
        for route in solution['routes']:
            route_distance = 0
            for i in range(len(route) - 1):
                route_distance += self.dist_matrix[route[i]][route[i+1]]
            total_distance += route_distance
        
        # Add large penalty for unassigned customers
        unassigned_penalty = len(solution['unassigned_customers']) * 10000
        
        return total_distance + unassigned_penalty

    def _update_pheromones(self, ant_solutions):
        """
        Update pheromone trails
        
        Args:
            ant_solutions: List of (solution, cost) pairs
        """
        # Evaporation
        self.pheromones *= (1 - self.evaporation)
        
        # Add new pheromones based on solution quality
        for solution, distance in ant_solutions:
            # Skip very poor solutions
            if distance > 1e9:
                continue
                
            # Calculate pheromone deposit
            deposit = 1.0 / distance
            
            # Add pheromones to edges used in solution
            for route in solution['routes']:
                for i in range(len(route) - 1):
                    self.pheromones[route[i]][route[i+1]] += deposit

    def _prepare_route_details(self):
        """
        Create detailed route information with city names
        
        Returns:
            List of route detail dictionaries
        """
        route_details = []
        
        for truck_idx, route in enumerate(self.best_solution['routes']):
            # Find customers in this route
            route_customers = []
            for cust_idx in self.best_solution['assigned_customers']:
                if self.customers[cust_idx]['city_index'] in route[1:-1]:
                    route_customers.append(self.customers[cust_idx])
            
            # Calculate route distance
            distance = 0
            for i in range(len(route) - 1):
                distance += self.dist_matrix[route[i]][route[i+1]]
            
            # Get city names
            city_names = [self.cities[idx] for idx in route]
            
            # Deduplicate city names
            unique_city_names = deduplicate_route_cities(city_names)
            
            # Calculate load
            load = sum(customer['demand'] for customer in route_customers)
            
            route_details.append({
                'truck_id': truck_idx + 1,
                'route_indices': route,
                'route_cities': unique_city_names,
                'customers': route_customers,
                'load': load,
                'distance': distance
            })
        
        return route_details

    def plot_convergence(self):
        """
        Create convergence plot
        
        Returns:
            Matplotlib figure
        """
        if not self.solution_history:
            return None
        
        iterations = [x[1] for x in self.solution_history]
        distances = [x[0] for x in self.solution_history]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(iterations, distances, 'b-', linewidth=2)
        
        ax.set_title('Convergence of ACO Algorithm', fontsize=14)
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('Best Solution Cost', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Add point markers at regular intervals
        marker_indices = np.linspace(0, len(iterations)-1, min(10, len(iterations))).astype(int)
        ax.plot([iterations[i] for i in marker_indices], 
                [distances[i] for i in marker_indices], 
                'bo', markersize=6)
        
        # Add horizontal line for final solution
        ax.axhline(y=distances[-1], color='r', linestyle='--', alpha=0.5)
        
        return fig

# Function to save uploaded truck icon
def save_uploaded_image(uploaded_file):
    """Save uploaded truck icon to temp file"""
    if uploaded_file is None:
        return None
    
    # Create temp directory if it doesn't exist
    temp_dir = 'temp'
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    
    # Save the file
    file_path = os.path.join(temp_dir, 'truck_icon.png')
    with open(file_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())
    
    return file_path

# Create route visualization
def create_route_visualization(cities, city_positions, route_details):
    """
    Create a static visualization of the optimized routes
    
    Args:
        cities: List of city names
        city_positions: Dictionary mapping cities to (x,y) coordinates
        route_details: List of route detail dictionaries
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Extract warehouses and destinations
    warehouse_cities = set()
    destination_cities = set()
    
    for rd in route_details:
        for city in rd['route_cities']:
            if rd['route_cities'].index(city) == 0:  # First city in route is warehouse
                warehouse_cities.add(city)
            else:
                destination_cities.add(city)
    
    # Plot warehouses with labels
    warehouse_x = [city_positions[city][0] for city in warehouse_cities if city in city_positions]
    warehouse_y = [city_positions[city][1] for city in warehouse_cities if city in city_positions]
    warehouse_scatter = ax.scatter(warehouse_x, warehouse_y, c='blue', s=120, marker='s', label='Warehouses')
    
    # Add warehouse labels
    for city in warehouse_cities:
        if city in city_positions:
            x, y = city_positions[city]
            ax.annotate(city, (x, y), xytext=(5, 5), textcoords='offset points', 
                       fontsize=9, fontweight='bold', color='darkblue')
    
    # Plot destinations
    dest_x = [city_positions[city][0] for city in destination_cities if city in city_positions]
    dest_y = [city_positions[city][1] for city in destination_cities if city in city_positions]
    ax.scatter(dest_x, dest_y, c='red', s=60, alpha=0.6, label='Destinations')
    
    # Plot routes with different colors
    colors = plt.cm.tab20(np.linspace(0, 1, len(route_details)))
    
    for i, rd in enumerate(route_details):
        # Skip empty routes
        if not rd['customers']:
            continue
            
        # Get route coordinates
        route_x = []
        route_y = []
        
        for city in rd['route_cities']:
            if city in city_positions:
                x, y = city_positions[city]
                route_x.append(x)
                route_y.append(y)
        
        # Skip if not enough points
        if len(route_x) < 2:
            continue
        
        # Plot route
        color = colors[i % len(colors)]
        ax.plot(route_x, route_y, '-', color=color, alpha=0.7, linewidth=2.5,
                label=f"Truck {rd['truck_id']} - {len(rd['customers'])} orders")
        
        # Highlight stops with markers
        ax.plot(route_x[1:-1], route_y[1:-1], 'o', color=color, markersize=8)
        
        # Add truck icon/label at starting point
        ax.text(route_x[0] + 0.01, route_y[0] + 0.01, f"🚚{rd['truck_id']}", 
                color=color, fontsize=12, fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.3'))
    
    # Customize plot
    ax.set_title('Optimized Delivery Routes', fontsize=16)
    ax.set_xlabel('X Position', fontsize=12)
    ax.set_ylabel('Y Position', fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Create custom legend with route information
    legend = ax.legend(loc='upper left', fontsize=10, title='Route Legend')
    legend.get_title().set_fontsize('12')
    
    # Add scale and improve aspect ratio
    ax.set_aspect('equal')
    
    return fig

# Data Upload Tab
with tabs[0]:
    st.markdown("<h2 class='sub-header'>Upload Required Data Files</h2>", unsafe_allow_html=True)
    
    st.markdown("<div class='info-box'>", unsafe_allow_html=True)
    st.write("Please upload the following required CSV files:")
    st.write("1. **distance.csv**: Contains Source, Destination, and Distance(M) columns")
    st.write("2. **order_large.csv** or **order_small.csv**: Contains order information")
    st.write("3. **Truck Icon**: (Optional) Upload a custom truck icon for the route visualization")
    st.markdown("</div>", unsafe_allow_html=True)
    
    # File upload section
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Upload Distance Data")
        distance_file = st.file_uploader("Upload distance.csv", type="csv")
        if distance_file is not None:
            try:
                distance_df = pd.read_csv(distance_file)
                required_cols = ['Source', 'Destination', 'Distance(M)']
                if all(col in distance_df.columns for col in required_cols):
                    st.success(f"✅ Distance data loaded: {len(distance_df)} routes")
                    st.session_state.distance_df = distance_df
                else:
                    missing = [col for col in required_cols if col not in distance_df.columns]
                    st.error(f"Missing columns: {', '.join(missing)}")
            except Exception as e:
                st.error(f"Error loading distance data: {str(e)}")
    
    with col2:
        st.subheader("Upload Order Data")
        order_option = st.radio("Select order dataset:", ["Large Orders", "Small Orders"])
        
        if order_option == "Large Orders":
            file_label = "order_large.csv"
        else:
            file_label = "order_small.csv"
            
        order_file = st.file_uploader(f"Upload {file_label}", type="csv")
            
        if order_file is not None:
            try:
                orders_df = pd.read_csv(order_file)
                required_cols = ['Order_ID', 'Source', 'Destination', 'Weight']
                missing = [col for col in required_cols if col not in orders_df.columns]
                
                if not missing:
                    # Preprocess orders
                    processed_orders = preprocess_orders(orders_df)
                    st.success(f"✅ Order data loaded and processed: {len(processed_orders)} orders")
                    st.session_state.orders_df = processed_orders
                    st.session_state.order_file_name = file_label
                else:
                    st.warning(f"Missing columns: {', '.join(missing)}. Some features may not work properly.")
            except Exception as e:
                st.error(f"Error processing orders: {str(e)}")
    
    with col3:
        st.subheader("Upload Truck Icon (Optional)")
        truck_icon_file = st.file_uploader("Upload custom truck icon", type=["png", "jpg", "jpeg"])
        
        if truck_icon_file is not None:
            try:
                # Save the uploaded icon
                icon_path = save_uploaded_image(truck_icon_file)
                st.session_state.truck_icon_path = icon_path
                
                # Display the icon
                st.image(truck_icon_file, caption="Custom Truck Icon", width=150)
                st.success("✅ Custom truck icon loaded")
            except Exception as e:
                st.error(f"Error saving truck icon: {str(e)}")
    
    # Show data preview if files are uploaded
    if 'distance_df' in st.session_state and 'orders_df' in st.session_state:
        st.subheader("Data Preview")
        
        tab1, tab2 = st.tabs(["Orders Data", "Distance Data"])
        
        with tab1:
            st.dataframe(st.session_state.orders_df.head(10))
            
            # Display order statistics
            st.write("**Order Statistics:**")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Orders", len(st.session_state.orders_df))
            with col2:
                if 'Weight' in st.session_state.orders_df.columns:
                    avg_weight = st.session_state.orders_df['Weight'].mean()
                    st.metric("Average Weight", f"{avg_weight:.2f}")
            with col3:
                st.metric("Unique Destinations", st.session_state.orders_df['Destination'].nunique())
        
        with tab2:
            st.dataframe(st.session_state.distance_df.head(10))
            
            # Display distance statistics
            st.write("**Distance Statistics:**")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Routes", len(st.session_state.distance_df))
            with col2:
                avg_dist = st.session_state.distance_df['Distance(M)'].mean()
                st.metric("Average Distance", f"{avg_dist:.2f}")
            with col3:
                unique_locations = pd.unique(st.session_state.distance_df[['Source', 'Destination']].values.ravel())
                st.metric("Unique Locations", len(unique_locations))

# Configure & Run Tab
with tabs[1]:
    st.markdown("<h2 class='sub-header'>Configure Algorithm & Run Optimization</h2>", unsafe_allow_html=True)
    
    if 'distance_df' not in st.session_state or 'orders_df' not in st.session_state:
        st.warning("Please upload all required data files in the Data Upload tab first")
    else:
        st.markdown("<div class='info-box'>", unsafe_allow_html=True)
        st.write(f"📊 Problem Size: {len(st.session_state.orders_df)} orders to be assigned to trucks")
        unique_sources = st.session_state.orders_df['Source'].nunique()
        unique_dests = st.session_state.orders_df['Destination'].nunique()
        st.write(f"📍 Locations: {unique_sources} unique sources, {unique_dests} unique destinations")
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Configuration options
        st.subheader("Vehicle Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            num_trucks = st.slider("Number of Trucks", min_value=1, max_value=100, value=20, 
                               help="Number of trucks available for deliveries")
            
        with col2:
            truck_capacity = st.slider("Truck Capacity (kg)", min_value=1000, max_value=30000, value=15000, step=1000,
                                  help="Maximum weight capacity per truck in kg")
        
        # Algorithm parameters
        st.subheader("Algorithm Parameters")
        col1, col2 = st.columns(2)
        
        with col1:
            num_ants = st.slider("Number of Ants", min_value=5, max_value=50, value=20, 
                              help="More ants can find better solutions but take longer")
            alpha = st.slider("Alpha (Pheromone Importance)", min_value=0.1, max_value=5.0, value=1.0, step=0.1,
                           help="Higher values give more importance to pheromone trails")
        
        with col2:
            beta = st.slider("Beta (Distance Importance)", min_value=0.1, max_value=5.0, value=3.0, step=0.1,
                          help="Higher values prioritize shorter distances")
            iterations = st.slider("Number of Iterations", min_value=5, max_value=100, value=30,
                                help="More iterations can improve the solution but take longer")
        
        evaporation = st.slider("Evaporation Rate", min_value=0.01, max_value=0.9, value=0.5, step=0.01,
                              help="Controls how quickly pheromone trails evaporate")
        
        # Run optimization
        if st.button("🚀 Start Optimization", type="primary"):
            st.subheader("Optimization Progress")
            
            # Create progress bar
            progress_bar = st.progress(0)
            
            try:
                # Load and prepare data
                dist_matrix, city_to_index, cities = load_distance_matrix(st.session_state.distance_df)
                customers = load_orders(st.session_state.orders_df, city_to_index)
                
                # Generate city positions for visualization
                city_positions = generate_city_positions(cities, customers)
                
                # Create solver
                solver = EnhancedACOSolver(
                    dist_matrix=dist_matrix,
                    customers=customers,
                    city_to_index=city_to_index,
                    cities=cities,
                    num_trucks=num_trucks,
                    truck_capacity=truck_capacity,
                    num_ants=num_ants,
                    num_iterations=iterations,
                    alpha=alpha,
                    beta=beta,
                    evaporation=evaporation,
                    progress_bar=progress_bar
                )
                
                # Run optimization
                with st.spinner("Running optimization... This may take a few minutes for large problems."):
                    results = solver.solve()
                
                # Store results in session state
                st.session_state.solver = solver
                st.session_state.results = results
                st.session_state.cities = cities
                st.session_state.city_positions = city_positions
                
                # Display success message
                assigned_orders = len(customers) - len(results['unassigned'])
                st.success(f"✅ Optimization Complete! Assigned {assigned_orders} of {len(customers)} orders to {results['route_details']} trucks.")
                
                # Show execution time
                st.info(f"Execution Time: {results['execution_time']:.2f} seconds")
                
                # Show convergence plot
                st.subheader("Convergence Plot")
                conv_fig = solver.plot_convergence()
                if conv_fig:
                    st.pyplot(conv_fig)
                
            except Exception as e:
                st.error(f"Error in optimization process: {str(e)}")
                import traceback
                st.code(traceback.format_exc())

# Results Tab
with tabs[2]:
    st.markdown("<h2 class='sub-header'>Results & Visualization</h2>", unsafe_allow_html=True)
    
    if 'results' not in st.session_state:
        st.warning("Please run the optimization first in the Configure & Run tab")
    else:
        # Access results
        results = st.session_state.results
        cities = st.session_state.cities
        city_positions = st.session_state.city_positions
        
        # Display solution summary
        st.markdown("<div class='success-box'>", unsafe_allow_html=True)
        st.subheader("Solution Summary")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Distance", f"{results['best_distance']:.2f}")
        with col2:
            # Count routes with assigned orders
            active_routes = [rd for rd in results['route_details'] if rd['customers']]
            st.metric("Trucks Used", len(active_routes))
        with col3:
            # Count total assigned orders
            total_assigned = sum(len(rd['customers']) for rd in results['route_details'])
            total_orders = total_assigned + len(results['unassigned'])
            st.metric("Orders Assigned", f"{total_assigned} / {total_orders}")
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Display route details
        st.subheader("Route Details")
        
        # Create tabs for routes with assigned orders
        active_routes = [rd for rd in results['route_details'] if rd['customers']]
        
        if active_routes:
            route_tabs = st.tabs([f"Truck {rd['truck_id']}" for rd in active_routes])
            
            for i, rd in enumerate(active_routes):
                with route_tabs[i]:
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        st.write(f"**Starting Location:** {rd['route_cities'][0]}")
                        st.write(f"**Total Load:** {rd['load']:.2f} kg")
                        st.write(f"**Route Distance:** {rd['distance']:.2f}")
                        st.write(f"**Number of Orders:** {len(rd['customers'])}")
                    
                    with col2:
                        # Show route sequence
                        st.write("**Route Sequence:**")
                        st.write(" → ".join(rd['route_cities']))
                        
                        # Show orders
                        st.write("**Orders in this route:**")
                        if len(rd['customers']) > 20:
                            # If too many orders, show a sample
                            st.write(f"{len(rd['customers'])} orders including:")
                            order_sample = [c['order_id'] for c in rd['customers'][:20]]
                            st.write(order_sample)
                            st.write("...")
                        else:
                            order_ids = [c['order_id'] for c in rd['customers']]
                            st.write(order_ids)
        else:
            st.warning("No routes with assigned orders found.")
        
        # Display unassigned orders
        if results['unassigned']:
            st.subheader("Unassigned Orders")
            unassigned_percent = (len(results['unassigned']) / (len(results['unassigned']) + total_assigned)) * 100
            st.warning(f"{len(results['unassigned'])} orders ({unassigned_percent:.1f}%) could not be assigned.")
            
            if len(results['unassigned']) > 100:
                # Show sample if too many
                st.write("Sample of unassigned orders:")
                st.write(results['unassigned'][:100])
            else:
                st.write(results['unassigned'])
            
            st.info("To reduce unassigned orders, try increasing the number of trucks or truck capacity.")
        
        # Display route visualization
        st.subheader("Route Visualization")
        st.write("Optimized delivery routes for your fleet:")
        
        # Create static visualization
        fig = create_route_visualization(cities, city_positions, results['route_details'])
        st.pyplot(fig)

# Add footer
st.markdown("""---
<p style='text-align: center;'>Advanced Vehicle Routing Optimization | Created with Streamlit and ACO</p>
""", unsafe_allow_html=True)