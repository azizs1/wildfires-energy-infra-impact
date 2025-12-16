import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx
from sklearn.cluster import DBSCAN
from shapely.geometry import Point
import networkx as nx

from config import IMG_DIR


# =============================================================================
# Wildfire Exposure Metrics (per-substation)
# =============================================================================

def compute_fire_density(infra_gdf, wf_gdf, buffer_km=10):
    """
    Count historical fires within buffer of each substation.
    
    Args:
        infra_gdf: GeoDataFrame of substations (EPSG:3310)
        wf_gdf: GeoDataFrame of wildfire perimeters (EPSG:3310)
        buffer_km: Buffer radius in kilometers
        
    Returns:
        GeoDataFrame with 'fire_count' column added
    """
    infra_gdf = infra_gdf.copy()
    
    # Buffer substations
    infra_buffered = infra_gdf.copy()
    infra_buffered["geometry"] = infra_gdf.geometry.buffer(buffer_km * 1000)
    
    # Spatial join to count overlapping fires
    joined = gpd.sjoin(infra_buffered, wf_gdf, predicate="intersects", how="left")
    fire_counts = joined.groupby("id").size().rename("fire_count")
    
    infra_gdf = infra_gdf.merge(fire_counts, on="id", how="left")
    infra_gdf["fire_count"] = infra_gdf["fire_count"].fillna(0).astype(int)
    
    return infra_gdf


def compute_burned_area_exposure(infra_gdf, wf_gdf, buffer_km=10, acres_col="BurnBndAc"):
    """
    Sum burned acreage within buffer of each substation.
    
    Args:
        infra_gdf: GeoDataFrame of substations (EPSG:3310)
        wf_gdf: GeoDataFrame of wildfire perimeters (EPSG:3310)
        buffer_km: Buffer radius in kilometers
        acres_col: Column name for burned acres in wf_gdf
        
    Returns:
        GeoDataFrame with 'burned_acres' column added
    """
    infra_gdf = infra_gdf.copy()
    
    infra_buffered = infra_gdf.copy()
    infra_buffered["geometry"] = infra_gdf.geometry.buffer(buffer_km * 1000)
    
    joined = gpd.sjoin(infra_buffered, wf_gdf, predicate="intersects", how="left")
    
    if acres_col in joined.columns:
        burned_area = joined.groupby("id")[acres_col].sum().rename("burned_acres")
    else:
        # Fallback: count fires if acres not available
        burned_area = joined.groupby("id").size().rename("burned_acres")
    
    infra_gdf = infra_gdf.merge(burned_area, on="id", how="left")
    infra_gdf["burned_acres"] = infra_gdf["burned_acres"].fillna(0)
    
    return infra_gdf


def compute_nearest_fire_distance(infra_gdf, wf_gdf):
    """Distance from each substation to nearest historical fire perimeter."""
    infra_gdf = infra_gdf.copy()
    
    # Use unary_union for faster distance calculation
    fire_union = wf_gdf.geometry.unary_union
    
    distances = infra_gdf.geometry.distance(fire_union) / 1000  # convert to km
    infra_gdf["nearest_fire_km"] = distances
    
    return infra_gdf


def compute_severity_exposure(infra_gdf, wf_gdf, buffer_km=10, severity_col="High_T"):
    """Sum high-severity burn area within buffer."""
    infra_gdf = infra_gdf.copy()
    
    infra_buffered = infra_gdf.copy()
    infra_buffered["geometry"] = infra_gdf.geometry.buffer(buffer_km * 1000)
    
    joined = gpd.sjoin(infra_buffered, wf_gdf, predicate="intersects", how="left")
    
    if severity_col in joined.columns:
        severity = joined.groupby("id")[severity_col].sum().rename("high_severity_acres")
    else:
        severity = pd.Series(0, index=infra_gdf["id"]).rename("high_severity_acres")
    
    infra_gdf = infra_gdf.merge(severity, on="id", how="left")
    infra_gdf["high_severity_acres"] = infra_gdf["high_severity_acres"].fillna(0)
    
    return infra_gdf


def compute_all_fire_metrics(infra_gdf, wf_gdf, buffer_km=10):
    """Compute all wildfire exposure metrics for substations."""
    gdf = infra_gdf.copy()
    
    print(f"Computing wildfire exposure metrics (buffer={buffer_km}km)...")
    
    gdf = compute_fire_density(gdf, wf_gdf, buffer_km)
    print(f"  - Fire density: mean={gdf['fire_count'].mean():.1f} fires/substation")
    
    gdf = compute_burned_area_exposure(gdf, wf_gdf, buffer_km)
    print(f"  - Burned area: mean={gdf['burned_acres'].mean():,.0f} acres/substation")
    
    gdf = compute_nearest_fire_distance(gdf, wf_gdf)
    print(f"  - Nearest fire: mean={gdf['nearest_fire_km'].mean():.1f} km")
    
    gdf = compute_severity_exposure(gdf, wf_gdf, buffer_km)
    print(f"  - High severity: mean={gdf['high_severity_acres'].mean():,.0f} acres/substation")
    
    return gdf

def cluster_substations(infra_graph, eps=5000, min_samples=3):
    nodes = []
    for nid, data in infra_graph.nodes(data=True):
        lon = data.get("longitude")
        lat = data.get("latitude")
        if lon is not None and lat is not None:
            nodes.append((nid, lon, lat))

    if not nodes:
        return None
    
    df = pd.DataFrame(nodes, columns=["id", "lon", "lat"])
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat), crs="EPSG:4326").to_crs(epsg=3310)
    coords = np.column_stack([gdf.geometry.x, gdf.geometry.y])
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(coords)
    gdf["cluster_label"] = labels
    return gdf


# =============================================================================
# Outage Impact Simulation
# =============================================================================

def simulate_cluster_outage(infra_graph, cluster_nodes, min_viable_size=5):
    """
    Simulate impact of losing all substations in a cluster.
    
    Args:
        infra_graph: NetworkX graph
        cluster_nodes: List of node IDs to remove
        min_viable_size: Minimum component size to be operational
        
    Returns:
        dict with impact statistics
    """
    H0 = infra_graph.copy()
    burned_in_graph = set(cluster_nodes) & set(H0.nodes())
    
    H1 = H0.copy()
    H1.remove_nodes_from(burned_in_graph)
    
    # Count substations stranded in non-viable fragments
    stranded = 0
    if H1.number_of_nodes() > 0:
        stranded = sum(len(c) for c in nx.connected_components(H1) if len(c) < min_viable_size)
    
    total_impacted = len(burned_in_graph) + stranded
    baseline_size = H0.number_of_nodes()
    pct_impacted = 0
    if baseline_size > 0:
        pct_impacted = total_impacted/baseline_size

    # make dict of degree
    # use this degree centrality to determine how much impact it has on overall network
    deg = dict(H0.degree())
    direct_degree_loss = sum(deg.get(n,1) for n in burned_in_graph)
    largest_component_size = max((len(c) for c in nx.connected_components(H1)), default=0)
    pct_giant = 0
    if baseline_size > 0:
        pct_giant = largest_component_size/baseline_size
    
    return {
        "directly_destroyed": len(burned_in_graph),
        "stranded_in_fragments": stranded,
        "total_impacted": total_impacted,
        "pct_of_network": pct_impacted,
        "direct_degree_loss": direct_degree_loss,
        "largest_component_size": largest_component_size,
        "pct_giant_component": pct_giant
    }


# =============================================================================
# Composite Risk Scoring
# =============================================================================

def normalize(series):
    """Normalize series to 0-1 range."""
    min_val, max_val = series.min(), series.max()
    if max_val > min_val:
        return (series - min_val) / (max_val - min_val)
    return pd.Series(0, index=series.index)


def compute_composite_cluster_risk(infra_graph, wf_gdf, eps=5000, min_samples=3, buffer_km=10, weight_outage=0.35, weight_betweenness=0.25, weight_fire=0.40,top_k=15):
    """Compute composite risk scores combining infrastructure and wildfire metrics."""
    
    # Ensure weights sum to 1
    total_weight = weight_outage + weight_betweenness + weight_fire
    weight_outage /= total_weight
    weight_betweenness /= total_weight
    weight_fire /= total_weight
    
    # Step 1: Cluster substations
    print("\n" + "="*60)
    print("COMPOSITE RISK ANALYSIS")
    print("="*60)
    
    infra_gdf = cluster_substations(infra_graph, eps, min_samples)
    if infra_gdf is None or infra_gdf.empty:
        print("No substations to analyze.")
        return pd.DataFrame()
    
    # Step 2: Compute wildfire exposure per substation
    wf_gdf_projected = wf_gdf.to_crs(epsg=3310) if wf_gdf.crs != "EPSG:3310" else wf_gdf
    infra_gdf = compute_all_fire_metrics(infra_gdf, wf_gdf_projected, buffer_km)
    
    # Step 3: Compute betweenness centrality
    betweenness = nx.betweenness_centrality(infra_graph, weight="importance")
    infra_gdf["betweenness"] = infra_gdf["id"].map(betweenness).fillna(0)
    
    # Step 4: Aggregate metrics by cluster
    print("\nAggregating metrics by cluster...")
    
    rows = []
    for cid in sorted(infra_gdf["cluster_label"].unique()):
        if cid == -1:  # Skip noise
            continue
        
        cluster_subs = infra_gdf[infra_gdf["cluster_label"] == cid]
        cluster_nodes = cluster_subs["id"].tolist()
        
        # Outage simulation
        impact = simulate_cluster_outage(infra_graph, cluster_nodes)
        
        # Aggregate wildfire metrics
        total_fire_count = cluster_subs["fire_count"].sum()
        total_burned_acres = cluster_subs["burned_acres"].sum()
        min_fire_distance = cluster_subs["nearest_fire_km"].min()
        total_severity = cluster_subs["high_severity_acres"].sum()
        
        # Average betweenness
        avg_betweenness = cluster_subs["betweenness"].mean()
        
        # Centroid for mapping
        centroid = cluster_subs.geometry.unary_union.centroid
        
        rows.append({
            "cluster_id": int(cid),
            "cluster_size": len(cluster_nodes),
            "centroid_x": centroid.x,
            "centroid_y": centroid.y,
            # Outage metrics
            "directly_destroyed": impact["directly_destroyed"],
            "stranded": impact["stranded_in_fragments"],
            "total_impacted": impact["total_impacted"],
            "pct_of_network": impact["pct_of_network"],
            "direct_degree_loss": impact["direct_degree_loss"],
            "largest_component_size": impact["largest_component_size"],
            "pct_giant_component": impact["pct_giant_component"],
            # Network metrics
            "avg_betweenness": avg_betweenness,
            # Wildfire metrics
            "fire_count": total_fire_count,
            "burned_acres": total_burned_acres,
            "nearest_fire_km": min_fire_distance,
            "high_severity_acres": total_severity,
        })
    
    df = pd.DataFrame(rows)
    
    if df.empty:
        print("No clusters found.")
        return df
    
    # Step 5: Normalize and compute composite score
    print("\nComputing composite risk scores...")
    
    # Normalize each component
    df["destroyed_norm"]   = normalize(df["directly_destroyed"])
    df["stranded_norm"]    = normalize(df["stranded"])
    df["degree_loss_norm"] = normalize(df["direct_degree_loss"])

    df["outage_norm"] = (
        0.25 * df["stranded_norm"] +
        0.20 * df["destroyed_norm"] +
        0.15 * df["degree_loss_norm"]
    )

    df["betweenness_norm"] = normalize(df["avg_betweenness"])
    
    # For wildfire, proximity risk is going to be modeled with exponential decay
    # fruteher the nearest fire is, the lower the risk here
    lambda_km = 10
    df["proximity_risk"] = np.exp(-df["nearest_fire_km"] / lambda_km)
    
    # Combine fire metrics: 60% burned area, 40% proximity
    df["fire_norm"] = (
        0.45 * normalize(df["burned_acres"]) +
        0.20 * normalize(df["high_severity_acres"]) +
        0.20 * df["proximity_risk"] +
        0.15 * normalize(df["fire_count"])
    )
    
    # Composite risk score
    df["composite_risk"] = (
        weight_outage * df["outage_norm"] +
        weight_betweenness * df["betweenness_norm"] +
        weight_fire * df["fire_norm"]
    )
    
    df = df.sort_values("composite_risk", ascending=False).reset_index(drop=True)
    
    # Step 6: Print summary
    print("\n" + "-"*60)
    print(f"TOP {top_k} HIGH-RISK CLUSTERS")
    print("-"*60)
    print(f"Weights: outage={weight_outage:.0%}, betweenness={weight_betweenness:.0%}, fire={weight_fire:.0%}")
    print()
    
    summary_cols = [
        "cluster_id", "cluster_size", "total_impacted", 
        "pct_of_network", "avg_betweenness", "burned_acres",
        "nearest_fire_km", "composite_risk"
    ]
    
    print(df.head(top_k)[summary_cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    
    # Step 7: Generate visualization
    plot_risk_map(df, infra_gdf, wf_gdf_projected, top_k)
    
    return df

def plot_risk_map(risk_df, infra_gdf, wf_gdf, top_k=10):
    """
    Generate a map showing high-risk clusters overlaid on wildfire history.
    
    Args:
        risk_df: DataFrame from compute_composite_cluster_risk
        infra_gdf: GeoDataFrame of substations with cluster labels
        wf_gdf: GeoDataFrame of wildfire perimeters
        top_k: Number of top clusters to highlight
    """
    fig, ax = plt.subplots(figsize=(14, 14))
    
    # Convert to web mercator for basemap
    wf_display = wf_gdf.to_crs(epsg=3857)
    infra_display = infra_gdf.to_crs(epsg=3857)
    
    # Plot wildfire perimeters
    wf_display.plot(
        ax=ax,
        column="BurnBndAc" if "BurnBndAc" in wf_display.columns else None,
        cmap="YlOrRd",
        alpha=0.3,
        legend=False
    )
    
    # Plot all substations as small gray dots
    infra_display.plot(ax=ax, color="gray", markersize=3, alpha=0.4)
    
    # Highlight top-k risk clusters
    top_clusters = risk_df.head(top_k)["cluster_id"].tolist()
    colors = plt.cm.Reds(np.linspace(0.4, 1, top_k))[::-1]
    
    for i, cid in enumerate(top_clusters):
        cluster_subs = infra_display[infra_display["cluster_label"] == cid]
        risk_score = risk_df[risk_df["cluster_id"] == cid]["composite_risk"].values[0]
        
        cluster_subs.plot(
            ax=ax,
            color=colors[i],
            markersize=7,
            alpha=0.8,
            label=f"#{i+1} Cluster {cid} (risk={risk_score:.3f})"
        )
    
    # Add basemap
    ctx.add_basemap(ax, source=ctx.providers.CartoDB.DarkMatter)
    
    ax.set_title(
        f"Wildfire-Infrastructure Risk Assessment\nTop {top_k} High-Risk Substation Clusters",
        fontsize=14,
        color="white",
        pad=20
    )
    
    ax.legend(
        loc="upper right",
        fontsize=9,
        facecolor="black",
        edgecolor="white",
        labelcolor="white"
    )
    
    plt.tight_layout()
    plt.savefig(IMG_DIR / "composite_risk_map.png", dpi=300, bbox_inches="tight")
    print(f"\nSaved risk map to {IMG_DIR / 'composite_risk_map.png'}")

def export_risk_report(risk_df, output_path=None):
    """
    Export risk analysis to CSV.
    
    Args:
        risk_df: DataFrame from compute_composite_cluster_risk
        output_path: Path for CSV output (default: IMG_DIR/risk_report.csv)
    """
    if output_path is None:
        output_path = IMG_DIR / "risk_report.csv"
    
    risk_df.to_csv(output_path, index=False)
    print(f"Exported risk report to {output_path}")
    
    return output_path