"""Functions for comparing bikeway datasets from OpenStreetMap and local agencies."""

import geopandas as gpd
import streetspace as sp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import osmnx as ox
import networkx as nx
from pathlib import Path
from collections import OrderedDict


def structure_bikeways_shapefile(shp_path, classification_function, name_column):
    """Restructure a bikeway shapefile with dummy variables for each bikeway type.

    Parameters
    ----------
    shp_path : :obj:`str`
        Path of bikeway shapefile
    
    classification_function : function
        Must operate row-wise on a Pandas DataFrame (e.g., df.apply(function, axis=1))

    name_column : :obj:`str`
        Shapefile column with street name

    Returns
    -------
    :class:`geopandas.GeoDataFrame`
        Restructured DataFrame
    """
    # Load shapefile
    edges = gpd.read_file(shp_path)
    crs = edges.crs
    # Apply classification function
    edges['bikeways'] = edges.apply(classification_function, axis=1)
    # Break bikeways into dummy variables
    edges_dummies = pd.get_dummies(edges['bikeways'])
    # Drop rows with no bikeways
    edges_dummies = edges_dummies[(
        (edges_dummies.T != 0).any())]
    edges = pd.concat(
        [edges_dummies, 
         edges[[name_column,'geometry']]], join='inner', axis=1)
    # Convert to GeoDataFrame
    edges = gpd.GeoDataFrame(edges, geometry='geometry', crs=crs)
    # Rename 'name' column
    edges = edges.rename(columns={name_column: 'name'})
    return edges


def summarize_facility(matches, summary_columns, summary_labels=['OSM', 'Local'],
    data_labels={1: 'Bikeway', 0: 'No Bikeway'}, agg_columns=['length'], 
    agg_funcs=[len, sum], agg_labels=['Edges', 'Total Length (km)'], 
    agg_postprocess=[lambda x: x.astype(int), lambda x: (x / 1000).round(1)]):
    """Summarize correspondence between local and OSM representations of the same bikeway type.

    Parameters
    ----------
    matches : :class:`pandas.DataFrame`
        DataFrame containing values to summarize.

    summary_columns : :obj:`list`
        List specifying columns in `matches` to summarize.

    summary_labels : :obj:`list`, optional, default = ['OSM', 'Local']
        List specifying names to use in place of original names of `summary_columns`.
        Must be the same length as `summary_columns`.

    data_labels : :obj:`dict`, optional, default = {1: 'Bikeway', 0: 'No Bikeway'}
        Dictionary specifying labels for values in `summary_columns`.
        Must have a key for each unique value in `summary_columns`.

    agg_columns : :obj:`list`, optional, default = ['length']
        List of columns in `matches` to summarize.

    agg_funcs : :obj:`list`, optional, default = [len, sum]
        List of functions to be applied to each column in `agg_columns`.
        Functions must operate on a list.

    agg_labels : :obj:`list`, optional, default = ['Edges', 'Total Length (km)']
        List of column labels corresponding to each function in `agg_funcs`.
        Length of `agg_labels` and `agg_funcs` must match.

    agg_postprocess : :obj:`list`, optional, default = [lambda x: x.astype(int), lambda x: (x / 1000).round(1)]
        List of functions for postprocessing the results of each function in `agg_funcs`.
        Functions must operate on a Pandas Series.
        Length of `agg_postprocess` and `agg_funcs` must match.

    Returns
    -------
    :class:`pandas.DataFrame`
        Multiindexed DataFrame summarizing specified columns
    """
    # Select columns to summarize
    summary = []
    # See whether each column exists in datset
    for column in summary_columns:
        if column in matches:
            # Extract column
            summary.append(matches[[column]].copy())
        else:
            # Make blank stand-in column
            summary.append(
                pd.Series([0]*len(matches), index=matches.index))
    summary = pd.concat(summary, axis=1)
    # Rename columns
    summary.columns = summary_labels
    # Reclassify values
    summary = summary.apply(lambda column: column.map(data_labels), axis=0)   
    # Add aggregation values
    summary = summary.merge(matches[agg_columns], left_index=True, right_index=True)   
    # Summarize the aggregation values by the summary columns
    summary = pd.pivot_table(
        summary,
        index=summary_labels, 
        values=agg_columns,
        aggfunc=agg_funcs)
    # Drop the second level of the column index
    summary.columns = summary.columns.droplevel(1)
    # Rename the columns   
    summary.columns = agg_labels
    # Postprocess the aggregations
    for column, process in zip(agg_labels, agg_postprocess):
        summary[column] = process(summary[column])
    return summary


# Function to compare local and osm bikeways
def summarize_bikeway_correspondance(matches, bikeway_labels):
    """Summarize correspondence between local and OSM representations of all bikeway types.
    
    Parameters
    ----------
    matches : :class:`pandas.DataFrame`
        DataFrame containing values to summarize.

    bikeway_labels : :obj:`dict` or :class:`collections.OrderedDict`
        Dictionary with keys specifying bikeway types.
        An OrderedDict is preferred because it maintains a consistent column ordering
            in the output table.

    Returns
    -------
    :obj:`list`
        List of outputs from `summarize_facility` (Pandas DataFrames)
    """
    # Initiate list to store all summaries
    summaries = []
    # Summarize each type of bikeway
    for bikeway, label in bikeway_labels.items():
        osm_column = bikeway + '_osm'
        local_column = bikeway + '_local'         
        summary = summarize_facility(
            matches, 
            summary_columns=[osm_column, local_column],
            summary_labels=['OSM', 'Local'],
            data_labels={1: 'Bikeway', 0: 'No Bikeway'},
            agg_labels=[(label, 'Edges'), (label, 'Length')])
        # Specify bikeway type as secondary column index
        summary.columns = pd.MultiIndex.from_tuples(summary.columns)
        summaries.append(summary)
    summaries = pd.concat(summaries, axis=1)
    return summaries


# Function plot comparision of osm and local bikeways
def plot_local_osm_comparison(summaries, city):
    """Plot comparisions between OSM and local bikeways
    
    Parameters
    ----------
    summaries : :class:`pandas.DataFrame`
        Output from `summarize_bikeway_correspondance`

    city : :obj:`str`
        Name of city to use in plot title
    """
    # Initialize a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,6))
    fig.suptitle((
        'Correspondance Between Local and '
        'OSM Bike Facilities in {}').format(city), size=15, y=1.02)
    # Adjust spacing between subplots 
    plt.subplots_adjust(wspace=.25)
    # Plot counts
    counts = summaries.iloc[:-1].xs('Edges', level=1, axis=1)
    counts.T.plot(
        kind='bar', stacked=True, title='Number of Edges',
        color=['g','b','r'], legend=False, ax=ax1)
    ax1.set_ylabel('Edges')
    ax1.set_frame_on(False)
    ax1.set_xticklabels(counts.columns, rotation=30, ha='right')
    # Plot lengths
    lengths = summaries.iloc[:-1].xs('Length', level=1, axis=1)
    lengths.T.plot(
        kind='bar', stacked=True, title='Cumulative Length', 
        color=['g','b','r'], legend=False, ax=ax2)
    ax2.set_ylabel('Kilometers')
    ax2.set_frame_on(False)
    ax2.set_xticklabels(lengths.columns, rotation=30, ha='right')
    # Make legend
    fig.subplots_adjust(bottom=0.22)
    handles, _ = ax1.get_legend_handles_labels()
    fig.legend(
        reversed(handles), 
        ['Local Only', 'OSM Only', 'Both Local & OSM'], 
        loc='lower center', ncol=3).draw_frame(False)
    plt.show()

def analyze_city(boundary, crs, local_edges):
    """Analyze correspondence between Local and OSM bikeways throughout a city.

    Parameters
    ----------
    boundary : :class:`shapely.geometry.Polygon`
        City boundary projected in WGS 84

    crs : epsg coordinate system 
        Local coordinate system in meters (e.g., UTM 10: {'init': 'epsg:26910'})

    local_edges : :class:`geopandas.GeoDataFrame`
        Output from `structure_bikeways_shapefile`

    Returns
    -------
    :obj:`tuple`
        * :class:`pandas.DataFrame`
            Output from `summarize_bikeway_correspondance`
        * :class:`geopandas.GeoDataFrame`
            OSM edges with OSM and local bikeway data attached
    """

    # Define OSM tag filter
    # Importantly, no paths with 'highway':'service' or 'highway':'motorway' tags will be returned
    tag_filter = ('["area"!~"yes"]["highway"!~"service|footway|motor|proposed|construction|abandoned|platform|raceway"]'
              '["bicycle"!~"no"]["access"!~"private"]')

    # Download the OSM data
    overpass_jsons = ox.osm_net_download(boundary, custom_filter=tag_filter)
    overpass_json = sp.merge_overpass_jsons(overpass_jsons)

    # Define bikeway columns and associated labels
    bikeway_types = [
        'off_street_path',
        'bike_blvd',
        'separated_bike_lane',
        'bike_lane',
        'shoulder',
        'sharrow',
        'bike_route']

    # Parse Overpass JSON into bikeway types
    overpass_parsed = sp.parse_osm_tags(
        overpass_json, bikeway_types, 
        true_value=1, false_value=0, none_value=np.nan)

    # Specify attributes to include in graph
    path_tags = (bikeway_types + ['highway'])
    ox.config(useful_tags_path=path_tags)
    # Convert json to graph
    G = ox.create_graph([overpass_parsed])   
    # Simply graph by removing all nodes that are not intersections or dead ends
    G = ox.simplify_graph(G, strict=True)
    # Make graph undirected
    G = nx.to_undirected(G)
    # Convert graph to geodataframes
    osm_edges = ox.graph_to_gdfs(G, nodes=False)
    # Project to local coordinate system 
    osm_edges = osm_edges.to_crs(crs)

    # Project city boundary to local coordinate system
    boundary, _ = ox.project_geometry(boundary, to_crs=crs) 

    # Constrain edges to those intersecting the city boundary polygon
    osm_edges = sp.gdf_intersecting_polygon(
        osm_edges, boundary)

    # Summarize bikeway values stored in lists
    osm_edges[bikeway_types] = osm_edges[bikeway_types].applymap(
        lambda x: sp.nan_any(x, 1, np.nan))

    # Idenfity largest available highway type
    def largest_highway(highways):
        # Specify highway order, 
        # largest (least bikable) to smallest (most bikable)
        highway_order = [ 
            'trunk',
            'primary',
            'secondary',
            'tertiary',
            'unclassified',
            'residential',
            'living_street',
            'cycleway']
        highways = sp.listify(highways)
        # Strip '_link' from tags
        highways = [x[:-5] if x[-5:] == '_link' else x for x in highways]
        # If list includes one of these tags, return the biggest one    
        ranked_highways = [x for x in highways if x in highway_order]    
        if len(ranked_highways) > 0:
            ranks = [highway_order.index(x) for x in ranked_highways]
            return highway_order[min(ranks)]
        # Otherwise, return 'other'
        else:
            return 'other'
    osm_edges['highway'] = osm_edges['highway'].apply(largest_highway)

    # Restrict edges to bikeable highway types
    bikable = [
        'primary',
        'secondary',
        'tertiary',
        'unclassified',
        'residential',
        'living_street',
        'cycleway']
    osm_edges = osm_edges[osm_edges['highway'].isin(bikable)].copy()
    			
    # Project local edges to local coordinate system
    local_edges = local_edges.to_crs(crs)
   
    # Restrict to local edges intersecting the city boundary
    local_edges = sp.gdf_intersecting_polygon(
        local_edges, boundary)

    # Match local edges to OSM edges
    analysis_columns = bikeway_types + ['geometry']
    # Match dataframes
    osm_matches = sp.match_lines_by_hausdorff(
        sp.select_columns(osm_edges, analysis_columns, suffix='_osm').rename(
            columns={'geometry_osm':'geometry'}),
        sp.select_columns(local_edges, analysis_columns, suffix='_local').rename(
            columns={'geometry_local':'geometry'}),
        constrain_target_features=True,
        distance_tolerance=20,
        azimuth_tolerance=20,
        match_fields=True)

    # Identify local and osm bikeway columns
    joint_bikeway_cols = [column for column in osm_matches.columns if 
                             any(bikeway in column for bikeway in bikeway_types)]

    # Reduce lists to a single single binary value
    osm_matches[joint_bikeway_cols] = osm_matches[joint_bikeway_cols].applymap(
        lambda x: sp.nan_any(x, 1, np.nan))

    # Drop records without a bikeway in either dataset
    osm_matches = osm_matches.dropna(how='all', subset=joint_bikeway_cols)

    # Reclassify NaN values as 0
    osm_matches = osm_matches.fillna(0)

    # Function fo calculate composite bikeways
    def composite_columns(matches, columns, suffix):
        # Select relevent columns
        relevent_columns = sp.select_columns(matches, [x + suffix for x in columns])
        # Assess whether there are any values of 1 across each row
        return relevent_columns.apply(lambda x: sp.nan_any(x, 1, 0), axis=1)

    # Define exclusive and shared bikeway types
    exclusive_bikeways = ['bike_lane','separated_bike_lane']
    shared_bikeways = ['bike_blvd','sharrow','bike_route']

    # Calculate composite of exclusive bikeways
    osm_matches['exclusive_bikeway_osm'] = composite_columns(
        osm_matches, exclusive_bikeways, '_osm')
    osm_matches['exclusive_bikeway_local'] = composite_columns(
        osm_matches, exclusive_bikeways, '_local')

    # Calculate composite of shared bikeways
    osm_matches['shared_bikeway_osm'] = composite_columns(
        osm_matches, shared_bikeways, '_osm')
    osm_matches['shared_bikeway_local'] = composite_columns(
        osm_matches, shared_bikeways, '_local')

    # Calculate composite of all bikeways
    osm_matches['any_bikeway_osm'] = composite_columns(
        osm_matches, bikeway_types, '_osm')
    osm_matches['any_bikeway_local'] = composite_columns(
        osm_matches, bikeway_types, '_local')

    # Calculate the length of each edge
    osm_matches['length'] = osm_matches['geometry'].apply(lambda x: x.length)

    # Add labels to bikeway types
    bikeway_labels = [
        'Off Street Path',
        'Bike Boulevard',
        'Separated Bike Lane',
        'Bike Lane',
        'Shoulder',
        'Sharrow',
        'Bike Route']
    bikeway_labels = OrderedDict(zip(bikeway_types, bikeway_labels))
    # Add labels for composite bikeway types
    bikeway_labels.update({'exclusive_bikeway':'Exclusive'})
    bikeway_labels.update({'shared_bikeway':'Shared'})
    bikeway_labels.update({'any_bikeway':'Any'})

    # Calculate summaries
    summaries = summarize_bikeway_correspondance(osm_matches, bikeway_labels)

    return summaries, osm_matches
