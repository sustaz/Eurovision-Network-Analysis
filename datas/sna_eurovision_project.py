from __future__ import division
# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from collections import Counter
import sys
import os

plt.rcParams["figure.figsize"] = (20,10)
from itertools import chain
import tqdm as tqdm
from colorthief import ColorThief
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.cluster.vq import whiten
warnings.filterwarnings('ignore')
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:80% !important; }</style>"))
from IPython.display import clear_output
clear_output()
import matplotlib.colors as mcolors

"""First we download the Eurovision 2021 votes dataset from , 
by the way we consider also the whole eurovision dataset that we have splitted
and preproced in an external preprocessing script, in order to have a more
readable notebook. Here we have our preprocess_dataset function:
"""

votes_jury = pd.read_csv('Eurovision_juryvotes_2021.csv')
votes_jury['Country (Voters (vertical), Finalists (horizontal))'] = ''
votes_tele = pd.read_csv('Eurovision_televotes_2021.csv').sort_values(by=['Country (Voters (vertical), Finalists (horizontal))'], ascending=True).reset_index()
votes_data = votes_jury + votes_tele

def preprocessing_dataset(data):
  data = data.drop('index', axis = 1).T
  data["Total"] = data.sum(axis=1)
  data["Total"][4] = 9999 #########
  data = data.sort_values(by=['Total'], ascending=False).T
  data = data.rename({'Country (Voters (vertical), Finalists (horizontal))': 'ind'}, axis='columns')
  data = data.set_index('ind')
  data.rename({9999: 'Total'}, axis=0, inplace=True)
  data = data.T
  data = data.reset_index()
  data = data.rename({'index': 'Country'}, axis='columns')
  data = data.reset_index()
  data = data.rename({'index': 'Rank'}, axis='columns')
  return data

votes_data = preprocessing_dataset(votes_data)

"""Here we display an example of the first five rows"""

votes_data.head()

df_copy = votes_data.copy()

print('Maximum number of votes given to a single nation from each voting nation: ',
      df_copy.drop({'Rank','Country','Total'}, axis = 1).max().sort_values(ascending=False))

"""In the `votes_data` dataframe we have the information of number of points each country recieved from other countries. We'll tranform in into edge-list of votes with ``melt`` transformation"""

votes_melted = votes_data.melt(
    ['Rank','Country','Total'],
    var_name = 'Source Country',value_name='points')

"""Now we have a very useful column called points, that is more manageable"""

votes_melted

votes_melted = votes_melted.drop(votes_melted[votes_melted['Source Country'] == votes_melted['Country']].index)

print('HOW MANY TIMES A NATION GOT THE MAXIMUM POINTS:\n',
       votes_melted[votes_melted['points'] == 24]['Country'].value_counts())

print('\nHOW MANY TIMES A NATION GOT THE MINIMUM POINTS:\n',
      votes_melted[votes_melted['points'] == 0]['Country'].value_counts())

print('\nHOW MANY TIMES A NATION GIVES THE MAXIMUM POINTS:\n', 
      votes_melted[votes_melted['points'] == 24]['Source Country'].value_counts())

print('\nHOW MANY TIMES A NATION GIVES THE MINIMUM POINTS:\n', 
      votes_melted[votes_melted['points'] == 0]['Source Country'].value_counts())

"""#NETWORK METRICS

By the 'points' column, we can define a clustering algortithm , basing on how
countries are similar wrt to the given votes.
"""

countries = pd.read_csv('countries.csv',index_col='country')

df_points_to = votes_melted.drop('Total', axis = 1).set_index('Source Country').sort_values('points', ascending = False)
df_points_to = df_points_to.reset_index()
dist = df_points_to.pivot(index='Source Country', columns='Country', values='points')
# vedi se riesci a clusterizzare per posizioni geografiche
dist.fillna(24, inplace=True)
df_scaled = whiten(dist.to_numpy())
mergings = linkage(df_scaled, method='complete') #method = 'ward'
plt.figure(figsize=(20,12))
dn = dendrogram(mergings, labels=np.array(dist.index), color_threshold=0.55*max(mergings[:,2]), leaf_rotation=90, leaf_font_size=14)

plt.show()

votes_melted[votes_melted['Source Country'] == 'Switzerland'].sort_values(by='points', ascending = False)

votes_melted = votes_melted.drop(votes_melted[votes_melted['points'] == 0].index)
votes_melted = votes_melted.drop('Rank', axis = 1)

"""Let's build a directed, weighted ``networkx`` graph from the edgelist in ``votes_melted``:"""

G = nx.from_pandas_edgelist(votes_melted, 
                            source='Source Country',
                            target='Country',
                            edge_attr='points',
                            create_using=nx.DiGraph())

nx.info(G)

length=nx.all_pairs_dijkstra_path_length(G)
length

"""And let's visualize it:"""

from networkx.algorithms.community import lukes_partitioning
tree = nx.bfs_tree(G, 'Greece')
lukes_partitioning(tree, max_size = 8)

from networkx.algorithms.distance_measures import periphery
from networkx.algorithms.core import core_number
diameter = nx.diameter(G.to_undirected())
#periphery(G)
core_num = core_number(G)
print({k: v for k, v in sorted(core_num.items(),reverse = True, key=lambda item: item[1])})

"""DENSITY MEASURE, A VALUE BTW 0 AND 1"""

from networkx.classes.function import density
from collections import Counter
total_nodes = list(G.nodes())
west_nodes = ['Italy', 'Spain', 'Portugal', 'France', 'Switzerland', 'Belgium', 'Germany', 'Netherlands', 'United Kingdom', 'Denmark', 'Greece', 'Sweden', 'Norway', 'Iceland', 'Ireland', 'San Marino', 'Finland', 'Malta']
est_nodes = list((Counter(total_nodes)-Counter(west_nodes)).elements())
Gsub_east = G.subgraph(est_nodes)
Gsub_west = G.subgraph(west_nodes)
print('G Density: ', density(G) , '\n')
print('Gsub_west Density: ', density(Gsub_west), '\n')
print('Gsub_east Density: ' , density(Gsub_east))

from networkx.algorithms import community
from networkx.algorithms.community.centrality import girvan_newman

"""communities_generator = community.girvan_newman(G)
top_level_communities = next(communities_generator)
next_level_communities = next(communities_generator)
sorted(map(sorted, next_level_communities))"""
comp = girvan_newman(G)
tuple(sorted(c) for c in next(comp))
import itertools
k = 20
for communities in itertools.islice(comp, k):
    print(tuple(sorted(c) for c in communities))

nx.draw_networkx(G,node_size=2500)

"""Every country is giving the same amount of points (out degree):"""

def orderpairs(a):  
    b = []
    leng = len(a)
    for i in range(0, leng):
        maximum = 0
        for j in range(0, len(a)):
            if maximum <= a[j][1]:
                ind = j
                maximum = a[j][1]
        b.append(a.pop(ind))
    return b

out_deg_tot = dict(orderpairs(list(G.out_degree(weight='points'))))
print('OUTPUT DEGREE:\n')
print(out_deg_tot)
print('\nINPUT DEGREE:\n')
in_deg_tot = dict(orderpairs(list(G.in_degree(weight='points'))))
print(in_deg_tot)

print('\nOUTPUT DEGREE east:\n')
out_deg_east = dict(orderpairs(list(Gsub_east.out_degree(weight='points'))))
print(out_deg_east)
print('\nINPUT DEGREE east:\n')
in_deg_east = dict(orderpairs(list(Gsub_east.in_degree(weight='points'))))
print(in_deg_east)

print('\nOUTPUT DEGREE west:\n')
out_deg_west = dict(orderpairs(list(Gsub_west.out_degree(weight='points'))))
print(out_deg_west)
print('\nINPUT DEGREE west:\n')
in_deg_west = dict(orderpairs(list(Gsub_west.in_degree(weight='points'))))
print(in_deg_west)

import numpy as np
def perc_dict(deg_tot, deg_slice):  
    mean_dict = {}
    for nation in deg_slice.keys():
        if deg_tot[nation] > 0:
            mean_dict[nation] = deg_slice[nation]/deg_tot[nation]
        else:
            mean_dict[nation] = 0
    return mean_dict
    
perc_out_east_east = perc_dict(out_deg_tot, out_deg_east)
perc_out_west_west = perc_dict(out_deg_tot, out_deg_west)
mean_out_west = np.array(list(perc_out_west_west.items()))[:,1].astype(float)
mean_out_east = np.array(list(perc_out_east_east.items()))[:,1].astype(float)

print('OUTPUT DEGREE ANALYSYS (votes given):\n')
print('Percentage of votes: \nFrom west to west: ', np.mean(mean_out_west)*100, '%\nFrom west to east: ', (1-np.mean(mean_out_west))*100, '%\n')
print('Percentage of votes: \nFrom east to west: ', (1-np.mean(mean_out_east))*100 , '%\nFrom east to east: ',np.mean(mean_out_east)*100, '%\n')

perc_in_east_east = perc_dict(in_deg_tot, in_deg_east)
perc_in_west_west = perc_dict(in_deg_tot, in_deg_west)
mean_in_west = np.array(list(perc_in_west_west.items()))[:,1].astype(float)
mean_in_east = np.array(list(perc_in_east_east.items()))[:,1].astype(float)

print('INPUT DEGREE ANALYSYS (votes recived):\n')
print('Percentage of votes: \nFrom west to west: ', np.mean(mean_in_west)*100, '%\nFrom west to east: ', (1-np.mean(mean_in_west))*100, '%\n')
print('Percentage of votes: \nFrom east to west: ', (1-np.mean(mean_in_east))*100 , '%\nFrom east to east: ',np.mean(mean_in_east)*100, '%\n')

"""Let's import the countries and assign to pos_geo the positions in terms of latitude and longitude"""

Gsub = G.edge_subgraph([(e[0],e[1]) for e in G.edges(data=True) if e[2]['points']>0])

plt.hist(dict(Gsub.degree()).values())

"""### However in degree is the one that determines the victory:

#NODE METRICS
"""

pos_geo = { node: 
           ( max(-10,min(countries.loc[node]['longitude'],55)), # fixing scale
             max(countries.loc[node]['latitude'],25)) #fixing scale
               for node in G.nodes() }

h = plt.hist(dict(G.in_degree(weight='points')).values())

deg_cen_points = dict(G.in_degree(weight='points'))
print({k: v for k, v in sorted(deg_cen_points.items(),reverse = True, key=lambda item: item[1])})
#{k:deg_cen_points[k] for k in deg_cen_points if deg_cen_points[k]==max(deg_cen_points.values())}

"""A drawing function for the centrality measures"""

def draw(G, pos, measures, measure_name):
    
    nodes = nx.draw_networkx_nodes(G, pos, node_size=1250, cmap=plt.cm.plasma, 
                                   node_color=list(measures.values()),
                                   nodelist=measures.keys())
    nodes.set_norm(mcolors.SymLogNorm(linthresh=0.01, linscale=1, base=10))
    labels = nx.draw_networkx_labels(G, pos)
    edges = nx.draw_networkx_edges(G, pos)

    plt.title(measure_name)
    plt.colorbar(nodes)
    plt.axis('off')
    plt.show()

"""We can analize the nodes with two approaches, considering their geographical positions or the Fruchterman positions"""

flag_pos = pos_geo # set to pos_geo if you want geo positions

if flag_pos == 'fruchterman': 
  pos = nx.layout.fruchterman_reingold_layout(G,k=1,weight = 'points',iterations=1000,scale = 2)
else:
  pos = pos_geo

"""PAGE RANK"""

page_rank = dict(nx.pagerank_numpy(G,weight='points'))
draw(G, pos , nx.pagerank_numpy(G), 'Page Rank')

"""BETWEENNESS CENTRALITY"""

between = dict(nx.betweenness_centrality(G,weight='points'))
draw(G, pos, nx.betweenness_centrality(G), 'Betweenness Centrality')

G.remove_node('Russia')
draw(G, pos, nx.betweenness_centrality(G), 'Betweenness Centrality')

"""DEGREE CENTRALITY"""

degree_centrality = nx.degree_centrality(G)
draw(G, pos, nx.degree_centrality(G), 'Degree Centrality')

"""EIGENVECTOR CENTRALITY"""

eigen_centrality = nx.eigenvector_centrality(G)
draw(G, pos, nx.eigenvector_centrality(G), 'Eigenvector Centrality')

"""HUBS AND AUTHORITY CENTRALITY """

h,a = nx.hits(G)
draw(G, pos, h, 'Graph HITS Hubs')
draw(G, pos, a, 'Graph HITS Authorities')

"""CLOSENESS CENTRALITY"""

closeness_centrality = nx.closeness_centrality(G)
draw(G, pos, nx.closeness_centrality(G), 'Closeness Centrality')

"""#FIND CLIQUES"""

G_un = G.to_undirected
cliques = nx.find_cliques(G_un)
from networkx.algorithms.community import k_clique_communities
#c = list(k_clique_communities(G_un, 2))

"""Degree Assortativity to measure the similarity of connections in the graph with respect to the node degree.

The assortativity coefficient is a number between −1 and 1, just as are correlation
coefficients. A large positive value means that connected nodes very much tend
share similar properties; a large negative value means that connected nodes tend
to possess very different properties; and a value close to 0 means no strong
association of the property values between connected nodes (where strength is
gauged in distance from what would be expected with a random null model). Note
that the assortativity coefficient is always about a specific property or variable of
the nodes in a network
"""

degree_assortativity = nx.degree_assortativity_coefficient(G, x='out', y='in')
print(degree_assortativity)

"""Let's assign to each coutry it's flag and position of the map"""

flags = {}
flag_color = {}
for node in tqdm.tqdm_notebook(G.nodes()):
    flags[node] = 'flags/'+(countries.loc[node]['country_code']).lower().replace(' ','')+'.png'   
    flag_color[node] =  ColorThief(flags[node]).get_color(quality=1)

def RGB(red,green,blue): 
    return '#%02x%02x%02x' % (red,green,blue)

"""Now we will draw all the parts one-by-one"""

ax=plt.gca()
fig=plt.gcf()
plt.axis('off')
plt.title('Eurovision Final Votes',fontsize = 24)

trans = ax.transData.transform
trans2 = fig.transFigure.inverted().transform

tick_params = {'top':'off', 'bottom':'off', 'left':'off', 'right':'off',
              'labelleft':'off', 'labelbottom':'off'} #flag grid params

styles = ['dotted','dashdot','dashed','solid'] # line styles


    
# draw edges
for e in G.edges(data=True):
    width = e[2]['points']/24 #normalize by max points
    style=styles[int(width*3)]
    if width>0.3: #filter small votes
        nx.draw_networkx_edges(G,pos,edgelist=[e],width=width, style=style, edge_color = RGB(*flag_color[e[0]]) )
        # in networkx versions >2.1 arrowheads can be adjusted

#draw nodes    
for node in G.nodes():      
    imsize = max((0.3*G.in_degree(node,weight='points')
                  /max(dict(G.in_degree(weight='points')).values()))**2,0.03)
    
    # size is proportional to the votes
    flag = mpl.image.imread(flags[node])
    
    (x,y) = pos[node]
    xx,yy = trans((x,y)) # figure coordinates
    xa,ya = trans2((xx,yy)) # axes coordinates
    
    country = plt.axes([xa-imsize/2.0,ya-imsize/2.0, imsize, imsize ])
    country.imshow(flag)
    country.axis('off')
    country.set_aspect('equal')
    country.tick_params(**tick_params)
    
fig.savefig('eurovision_map.png')