import numpy as np
import matplotlib.pyplot as plt
import random
from PIL import Image, ImageDraw
import math, random
from typing import List, Tuple
from scipy.spatial import Voronoi, voronoi_plot_2d
from matplotlib import colors as mcolors
import sys
import json
from matplotlib import cm
m=0
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--name', type=int, required=True)
args = parser.parse_args()
colors = dict(mcolors.TABLEAU_COLORS, **mcolors.BASE_COLORS)

# Sort colors by hue, saturation, value and name.
by_hsv = sorted((tuple(mcolors.rgb_to_hsv(mcolors.to_rgba(color)[:3])), name)
                for name, color in colors.items())
sorted_names = ["#A19E74","#B18463", "#E8D7FF","#FEFBA3", "#292A39","#FFD2FC","#68D1CC","#FC696B","#D9BA8B","#232B33","#D3E7D0","#39272F","#33443E","#7B813D","#E980FC","#D65E2E","#D57C59","#8E838C","#3F3052" ,"#043E5F","#8CD0A1","#C1DBAE","#B96AC9","#231B1B","#640D0E" ,"#D3B675" ,"#82A07E" ,"#B89C6F" ]

sorted_name = [[0.1, 0.1, 0.1], [0.15, 0.15, 0.15],[0.2, 0.2, 0.2], [0.25, 0.1, 0.1], [0.30, 0.15, 0.15],[0.35, 0.1, 0.1],[0.40, 0.1, 0.1], [0.45, 0.15, 0.15],[0.5, 0.1, 0.1],[0.55, 0.1, 0.1], [0.60, 0.15, 0.15],[0.65, 0.1, 0.1]]
def voronoi_finite_polygons_2d(vor, radius=None):
    min_x = sys.maxsize
    max_x =-sys.maxsize
    min_y = sys.maxsize
    max_y= -sys.maxsize
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.

    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.

    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.

    """
    

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1] # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius
            min_x = min_x if far_point[0]>min_x else far_point[0]
            min_y = min_y if far_point[1]>min_y else far_point[1]
            max_x = max_x if far_point[0]<max_x else far_point[0]
            max_y = max_y if far_point[1]<max_y else far_point[1]
           
            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices), min_x, min_y, max_x, max_y

# make up data points
"""np.random.seed(1234)
points = np.random.rand(15, 2)

# compute Voronoi tesselation
vor = Voronoi(points)

# plot
regions, vertices = voronoi_finite_polygons_2d(vor)

# colorize
for region in regions:
    polygon = vertices[region]
    plt.fill(*zip(*polygon), alpha=0.4)

plt.plot(points[:,0], points[:,1], 'ko')
plt.xlim(vor.min_bound[0] - 0.1, vor.max_bound[0] + 0.1)
plt.ylim(vor.min_bound[1] - 0.1, vor.max_bound[1] + 0.1)

plt.show() """

for i in range(1):
    points = []
    num_corners = random.randint(3, 15)
    for c in range(num_corners):
        cx = random.randint(0, 1024)
        cy = random.randint(0, 1024)
        points.append([cx, cy])
    points = np.array(points)
    vor = Voronoi(points)
    regions, vertices , min_x, min_y, max_x,max_y = voronoi_finite_polygons_2d(vor)
    a=0
    
    img = Image.new("RGB", (4084, 4084), (255))
    draw = ImageDraw.Draw(img)



    for region in regions:
        polygon = vertices[region]
        
        
        color=  sorted_names[a]
        #h = color.lstrip('#')
        #rgb = list(int(h[i:i+2], 16)/256.0 for i in (0, 2, 4))
        #print(color)
       
        
        #plt.fill(*zip(*polygon), color=color)
        poly =[]
        for p in polygon:
            
            poly.append( (p[0]+abs(min_x))*4084/(max_x-min_x))

            poly.append( (p[1]+abs(min_y))*4084/(max_y-min_y))
        #print(poly)
        draw.polygon(poly, fill=color, outline=None)
   

       
       

        a+=1
    x1 = (vor.min_bound[0] - 0.1 +abs(min_x) )*4084/(max_x-min_x)
    x2 =(vor.max_bound[0] + 0.1+abs(min_x) )*4084/(max_x-min_x)
    y1 =(vor.min_bound[1] - 0.1+abs(min_y))*4084/(max_y-min_y)
    y2= (vor.max_bound[1] + 0.1+abs(min_y))*4084/(max_y-min_y)
    #img.save("img1_{}.png".format(i))
    im1 =img.crop((x1,y1,x2,y2))
    #im1=im1.resize((1024,1024))
    im1.save("images_test/img2_{}.png".format(args.name))
   
   

    
    #plt.plot(points[:,0], points[:,1], 'ko')
    #plt.xlim(vor.min_bound[0] - 0.1, vor.max_bound[0] + 0.1)
    #plt.ylim(vor.min_bound[1] - 0.1, vor.max_bound[1] + 0.1)
    #plt.xlim(0,200)
    #plt.ylim(0, 200)
    #print(vor.vertices)
    #fig = plt.gcf()
    
   
   
    
   
    

   
    img=im1
    # img=.open("img.png")
    a,b =img.size[0], img.size[1]
    #print(a,b)
    #img=np.array(img)
    #img =img[10:a-10,10:b-10,0]
    #img = [[(float("{:.2f}".format(k))) for k in a] for a in img]
    
    img=np.array(img)
    #print(a,b)
    #print(len(np.unique(img,axis=2)))
   
    img =img[10:b-10,10:a-10,0]
    h, w =b-20 ,a-20
    #print(len(np.unique(img,axis=2)))

   
    ss= set()
    for s in range(b-21):
        for t in range(a-21):
            ss.add(img[s][t])
    a=1
    
    data ={}
    data[0] = [h,w]
    
    import cv2 as cv
    for k in ss:
        piece = img.copy()
        piece[piece!=k]=0
        piece[piece==k]= (a+2)*(a+10)
        #print(piece.shape , h ,w )
        room_mask = (piece/((a+2)*(a+10)) * 255).astype(np.uint8)

        contours, _ = cv.findContours(room_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
       
        contours = contours[0]
        epsilon = 0.01*cv.arcLength(contours,True)
        approx = cv.approxPolyDP(contours,epsilon,True)
        #import pdb; pdb.set_trace()


        
       
        #cv.drawContours(room_mask, [contours], 0, (100,100,100), 3)
        #cv.imwrite("hey.png",room_mask)

        #print(np.array(contours))
       
        #plt.show()
        
        
        data[a]=approx.tolist()
        piece[piece==(a+2)*(a+10)]= 255 #(a+2)*(a+10)

        #p_im = Image.fromarray(np.uint8(cm.gist_earth(piece)*255))
        #draw = ImageDraw.Draw(p_im)
        #for c in contours:
            #print(c[0])
        #    draw.point(c[0], fill="blue")
        
        #piece[piece==k]= (a+2)*(a+10)
        
        #print(np.where(np.array(p_im)==0))

        #p_im.save("images/img_{}_piece_{}.png".format(i+m,a))
        a+=1
   
    with open('jsons_test/data_{}.json'.format(args.name), 'w') as f:
            json.dump(data, f)

   



   


    
    
