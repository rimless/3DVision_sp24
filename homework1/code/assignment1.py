## powered by xpd1
## on 2024-02-22

import trimesh
from trimesh import graph, grouping
from trimesh.geometry import faces_to_edges
import numpy as np
from itertools import zip_longest
from typing import List, Dict, Tuple
import functools
import heapq
import time



### def subdivision_loop(mesh):
def subdivision_loop(mesh, iterations):
    """
    Apply Loop subdivision to the input mesh for the specified number of iterations.
    :param mesh: input mesh
    :param iterations: number of iterations
    :return: mesh after subdivision
    
    Overall process:
    Reference: https://github.com/mikedh/trimesh/blob/main/trimesh/remesh.py#L207
    1. Calculate odd vertices.
      Assign a new odd vertex on each edge and
      calculate the value for the boundary case and the interior case.
      The value is calculated as follows.
          v2
        / f0 \\        0
      v0--e--v1      /   \\
        \\f1 /     v0--e--v1
          v3
      - interior case : 3:1 ratio of mean(v0,v1) and mean(v2,v3)
      - boundary case : mean(v0,v1)
    2. Calculate even vertices.
      The new even vertices are calculated with the existing
      vertices and their adjacent vertices.
        1---2
       / \\/ \\      0---1
      0---v---3     / \\/ \\
       \\ /\\/    b0---v---b1
        k...4
      - interior case : (1-kB):B ratio of v and k adjacencies
      - boundary case : 3:1 ratio of v and mean(b0,b1)
    3. Compose new faces with new vertices.
    
    # The following implementation considers only the interior cases
    # You should also consider the boundary cases and more iterations in your submission
    """
    
    vertices, faces = mesh.vertices, mesh.faces # [N_vertices, 3] [N_faces, 3]
    for iter2 in range(iterations):

        # prepare geometry for the loop subdivision
        
        edges, edges_face = faces_to_edges(faces, return_index=True) # [N_edges, 2], [N_edges]
        edges.sort(axis=1)
        unique, inverse = grouping.unique_rows(edges)
        
        # split edges to interior edges and boundary edges
        edge_inter = np.sort(grouping.group_rows(edges, require_count=2), axis=1)
        edge_bound = grouping.group_rows(edges, require_count=1)
        
        # set also the mask for interior edges and boundary edges
        edge_bound_mask = np.zeros(len(edges), dtype=bool)
        edge_bound_mask[edge_bound] = True
        edge_bound_mask = edge_bound_mask[unique]
        edge_inter_mask = ~edge_bound_mask
        
        ###########
        # Step 1: #
        ###########
        # Calculate odd vertices to the middle of each edge.
        odd = vertices[edges[unique]].mean(axis=1) # [N_oddvertices, 3]
        
        # connect the odd vertices with even vertices
        # however, the odd vertices need further updates over it's position
        # we therefore complete this step later afterwards.
        
        ###########
        # Step 2: #
        ###########
        # find v0, v1, v2, v3 and each odd vertex
        # v0 and v1 are at the end of the edge where the generated odd vertex on
        # locate the edge first
        e = edges[unique[edge_inter_mask]]
        # locate the endpoints for each edge
        e_v0 = vertices[e][:, 0]
        e_v1 = vertices[e][:, 1]
        
        # v2 and v3 are at the farmost position of the two triangle
        # locate the two triangle face
        edge_pair = np.zeros(len(edges)).astype(int)
        edge_pair[edge_inter[:, 0]] = edge_inter[:, 1]
        edge_pair[edge_inter[:, 1]] = edge_inter[:, 0]
        opposite_face1 = edges_face[unique]
        opposite_face2 = edges_face[edge_pair[unique]]
        # locate the corresponding edge
        e_f0 = faces[opposite_face1[edge_inter_mask]]
        e_f1 = faces[opposite_face2[edge_inter_mask]]
        # locate the vertex index and vertex location
        e_v2_idx = e_f0[~(e_f0[:, :, None] == e[:, None, :]).any(-1)]
        e_v3_idx = e_f1[~(e_f1[:, :, None] == e[:, None, :]).any(-1)]
        e_v2 = vertices[e_v2_idx]
        e_v3 = vertices[e_v3_idx]
        
        # update the odd vertices based the v0, v1, v2, v3, based the following:
        # 3 / 8 * (e_v0 + e_v1) + 1 / 8 * (e_v2 + e_v3)
        odd[edge_inter_mask] = 0.375 * e_v0 + 0.375 * e_v1 + e_v2 / 8.0 + e_v3 / 8.0

        ###########
        # Step 3: #
        ###########
        # find vertex neightbors for even vertices and update accordingly
        neighbors = graph.neighbors(edges=edges[unique], max_index=len(vertices))
        # convert list type of array into a fixed-shaped numpy array (set -1 to empties)
        neighbors = np.array(list(zip_longest(*neighbors, fillvalue=-1))).T
        # if the neighbor has -1 index, its point is (0, 0, 0), so that it is not included in the summation of neighbors when calculating the even
        vertices_ = np.vstack([vertices, [0.0, 0.0, 0.0]])
        # number of neighbors
        k = (neighbors + 1).astype(bool).sum(axis=1)
        
        # calculate even vertices for the interior case
        beta = (40.0 - (2.0 * np.cos(2 * np.pi / k) + 3) ** 2) / (64 * k)
        even = (
            beta[:, None] * vertices_[neighbors].sum(1)
            + (1 - k[:, None] * beta[:, None]) * vertices
        )
        
        ############
        # Step 1+: #
        ############
        # complete the subdivision by updating the vertex list and face list
        

        # if edge_bound_mask.any():                                        ###   We handle the bound vertices now
        #     bound_maskvrt = np.zeros(len(vertices), dtype=bool);
        #     bound_maskvrt[np.unique(edges[unique][~edge_inter_mask])] = True;
        #     neig_bound = neighbors[bound_maskvrt];
        #     bmaskvrt2 = bound_maskvrt[neighbors[bound_maskvrt]];
        #     neig_bound[~bmaskvrt2] = -1;
        #     ###print(neig_bound.shape,bound_maskvrt.shape,neighbors[bound_maskvrt].shape,bound_maskvrt[neighbors[bound_maskvrt]].shape,neig_bound[~bound_maskvrt[neighbors[bound_maskvrt]]].shape)
        #     even[bound_maskvrt] = (vertices[bound_maskvrt] * 0.75 + vertices_[neig_bound].sum(axis=1) * 0.125);

        if edge_bound_mask.any():
            # boundary vertices from boundary edges
            vrt_bound_mask = np.zeros(len(vertices), dtype=bool)
            vrt_bound_mask[np.unique(edges[unique][~edge_inter_mask])] = True
            # boundary neighbors
            boundary_neighbors = neighbors[vrt_bound_mask]
            boundary_neighbors[~vrt_bound_mask[neighbors[vrt_bound_mask]]] = -1

            even[vrt_bound_mask] = (
                vertices_[boundary_neighbors].sum(axis=1) / 8.0
                + (3.0 / 4.0) * vertices[vrt_bound_mask]
            )



        # the new faces with odd vertices
        odd_idx = inverse.reshape((-1, 3)) + len(vertices)
        new_faces = np.column_stack(
            [
                faces[:, 0],
                odd_idx[:, 0],
                odd_idx[:, 2],
                odd_idx[:, 0],
                faces[:, 1],
                odd_idx[:, 1],
                odd_idx[:, 2],
                odd_idx[:, 1],
                faces[:, 2],
                odd_idx[:, 0],
                odd_idx[:, 1],
                odd_idx[:, 2],
            ]
        ).reshape((-1, 3)) # [N_face*4, 3]

        # stack the new even vertices and odd vertices
        new_vertices = np.vstack((even, odd)) # [N_vertex+N_edge, 3]
        
        vertices = new_vertices;                  ###
        faces = new_faces;                        ###


    return trimesh.Trimesh(new_vertices, new_faces)

class VTXS:
    def __init__(self, ind, position) -> None:
        self.ind = ind
        self.position = position
        self.adjacent_vertices = set()
        self.Quad = None

    def add_an_adjacent(self, vid):
        self.adjacent_vertices.add(vid)

    def add_face_quadric(self, faceQuad):
        if self.Quad is None:
            self.Quad = faceQuad
        else:
            self.Quad += faceQuad


@functools.total_ordering
class EDGES:
    def __init__(self, v1, v2) -> None:
        self.v1 = v1
        self.v2 = v2
        self.adjacent_faces = set()
        self.Quad = None
        self.error_min = None
        self.error_vertices_min = None
    
    def add_an_adjacent(self, idFaces):
        self.adjacent_faces.add(idFaces)
    
    def __gt__(self, other):
        if (self.error_min is None or other.error_min is None): 
            return True;
        return self.error_min > other.error_min
    
    def __eq__(self, __value) -> bool:
        return self.error_min == __value.error_min

class GRAPH:
    def __init__(self, vertices, faces) -> None:
        self.num_vertices = vertices.shape[0]
        self.num_faces = faces.shape[0]

        self.vertices: List[VTXS] = [VTXS(i, vertices[i]) for i in range(self.num_vertices)]
        self.Edges: Dict[Tuple[int], EDGES] = dict()

        for i in range(self.num_faces):
            vertices = faces[i]
            vertices = np.sort(vertices)
            v1, v2, v3 = vertices  # v1 < v2 < v3
            
            self.vertices[v1].add_an_adjacent(v2)
            self.vertices[v1].add_an_adjacent(v3)
            self.vertices[v2].add_an_adjacent(v1)
            self.vertices[v2].add_an_adjacent(v3)
            self.vertices[v3].add_an_adjacent(v1)
            self.vertices[v3].add_an_adjacent(v2)

            if (v1, v2) not in self.Edges:
                self.Edges[(v1, v2)] = EDGES(v1, v2)
           
            if (v2, v3) not in self.Edges:
                self.Edges[(v2, v3)] = EDGES(v2, v3)
            
            if (v1, v3) not in self.Edges:
                self.Edges[(v1, v3)] = EDGES(v1, v3)

            self.Edges[(v1, v2)].add_an_adjacent(i)
            self.Edges[(v2, v3)].add_an_adjacent(i)
            self.Edges[(v1, v3)].add_an_adjacent(i)
        self.faces: List = faces.tolist()

    def edgeInter(self, v1, v2):
        if v1 > v2:
            v1, v2 = v2, v1
        return len(self.Edges[(v1, v2)].adjacent_faces) == 2
    
    def verticesOppo(self, idFaces, v1, v2):
        vertices = self.faces[idFaces]
        for v in vertices:
            if v != v1 and v != v2:
                return v
        return -1
    
    def verticesOppoAll(self, v1, v2):
        if v1 > v2:
            v1, v2 = v2, v1
        tmp = [self.verticesOppo(idFaces, v1, v2) for idFaces in self.Edges[(v1, v2)].adjacent_faces]
        assert len(tmp) == 2
        return tmp[0], tmp[1]
    
    def verticesNeighAll(self, v):
        return self.vertices[v].adjacent_vertices


def K_calculation(v1, v2, v3):
    e1 = v2 - v1
    e2 = v3 - v1
    n = np.cross(e1, e2)
    n = n / np.linalg.norm(n)
    a, b, c = n[0], n[1], n[2]
    d = -np.dot(n, v1)
    K = np.array([a**2, a*b, a*c, a*d, b**2, b*c, b*d, c**2, c*d, d**2])
    return K

def errorMinCalculation(K):
    KmatrixOriginal = np.array([[K[0], K[1], K[2], K[3]],
                      [K[1], K[4], K[5], K[6]],
                      [K[2], K[5], K[7], K[8]],
                      [K[3], K[6], K[8], K[9]]])
    k_mat = KmatrixOriginal.copy()
    k_mat[3, 3] = 1
    k_mat[3, :3] = 0
    verticesMin = np.linalg.inv(k_mat) @ np.array([[0, 0, 0, 1]]).T # (4, 1)
    error_min = verticesMin.T @ KmatrixOriginal @ verticesMin
    error_min = error_min[0, 0]
    verticesMin = verticesMin.squeeze()[:3]
    return verticesMin, error_min


def simplify_quadric_error(mesh, face_count=1):
    """
    Apply quadratic error mesh decimation to the input mesh until the target face count is reached.
    :param mesh: input mesh
    :param face_count: number of faces desired in the resulting mesh.
    :return: mesh after decimation
    """
    vertices = mesh.vertices
    faces = mesh.faces

    gph = GRAPH(vertices, faces)
    dictionFace = dict()
    for i in range(len(faces)):
        V1_idx, V2_idx, V3_idx = faces[i]
        v1, v2, v3 = vertices[V1_idx], vertices[V2_idx], vertices[V3_idx]
        K = K_calculation(v1, v2, v3)
        gph.vertices[V1_idx].add_face_quadric(K)
        gph.vertices[V2_idx].add_face_quadric(K)
        gph.vertices[V3_idx].add_face_quadric(K)

        dictionFace[V1_idx] = dictionFace.get(V1_idx, []) + [i]
        dictionFace[V2_idx] = dictionFace.get(V2_idx, []) + [i]
        dictionFace[V3_idx] = dictionFace.get(V3_idx, []) + [i]
    
    for V1_idx, V2_idx in gph.Edges.keys():
        if gph.edgeInter(V1_idx, V2_idx):
            v1, v2 = gph.vertices[V1_idx], gph.vertices[V2_idx]
            e = gph.Edges[(V1_idx, V2_idx)]
            K = v1.Quad + v2.Quad
            e.Quad = K
            e.error_vertices_min, e.error_min = errorMinCalculation(K)
        else:
            e = gph.Edges[(V1_idx, V2_idx)]


    edgeHeap = list(gph.Edges.values())
    faceCurrent = len(gph.faces)
    heapq.heapify(edgeHeap)
    while faceCurrent > face_count:
        e = heapq.heappop(edgeHeap)
        if e.Quad is None: # no longer valid after collapse
            continue
        V1_idx, V2_idx = e.v1, e.v2
        v1, v2 = gph.vertices[V1_idx], gph.vertices[V2_idx]
        if v1.position is None or v2.position is None:  # already collapsed
            continue

        v1.position = e.error_vertices_min
        v2.position = None
        v1.Quad = e.Quad
        v2.Quad = None

        for vid in v2.adjacent_vertices:
            KEY = (V2_idx, vid) if V2_idx < vid else (vid, V2_idx)
            gph.Edges.pop(KEY)
            gph.vertices[vid].adjacent_vertices.remove(V2_idx)
            if vid != V1_idx:
                gph.vertices[vid].adjacent_vertices.add(V1_idx)
        
        v1.adjacent_vertices = v1.adjacent_vertices.union(v2.adjacent_vertices)
        v1.adjacent_vertices.remove(V1_idx)
        for vid in v1.adjacent_vertices:
            KEY = (V1_idx, vid) if V1_idx < vid else (vid, V1_idx)
            if KEY in gph.Edges:
                edgePrevious = gph.Edges[KEY]
                edgePrevious.Quad = None

            edgeNew = EDGES(KEY[0], KEY[1])
            edgeNew.Quad = v1.Quad + gph.vertices[vid].Quad
            edgeNew.error_vertices_min, edgeNew.error_min = errorMinCalculation(edgeNew.Quad)
            gph.Edges[KEY] = edgeNew
            heapq.heappush(edgeHeap, edgeNew)

        # update faces
        faceDeleted = []
        for idFaces in dictionFace[V1_idx]:
            if V2_idx in gph.faces[idFaces]:
                faceCurrent -= 1
                faceDeleted.append(idFaces)

                verticesAdj = [v for v in gph.faces[idFaces] if v != V2_idx and v != V1_idx][0]
                dictionFace[verticesAdj].remove(idFaces)     
        for idFaces in faceDeleted:
            dictionFace[V1_idx].remove(idFaces)
            gph.faces[idFaces] = None

        for idFaces in dictionFace[V2_idx]:
            f = gph.faces[idFaces]
            if f is None:
                continue
                
            dictionFace[V1_idx].append(idFaces)
            for i, vid in enumerate(f):
                if vid == V2_idx:
                    gph.faces[idFaces][i] = V1_idx

        dictionFace.pop(V2_idx)

    vertices = np.stack([v.position for v in gph.vertices if v.position is not None])
    ExistVertices = np.array([v.position is not None for v in gph.vertices], dtype=bool)
    VerticesNew = ExistVertices.astype(int).cumsum() - ExistVertices
    faces = np.array([[VerticesNew[i] for i in f if ExistVertices[i]] for f in gph.faces if f is not None])
    print(vertices.shape, faces.shape)

    mesh = trimesh.Trimesh(vertices, faces)
    return mesh





if __name__ == '__main__':

    
    # Load mesh and print information
    # mesh = trimesh.load_mesh('assets/cube.obj')
    #mesh = trimesh.creation.box(extents=[1, 1, 1])
    mesh = trimesh.load_mesh('bunny.obj');###generic_neutral_mesh_FaceOnly_tri.obj')
    print(f'Mesh Info: {mesh}')
    
    # apply loop subdivision over the loaded mesh
    # mesh_subdivided = mesh.subdivide_loop(iterations=1)
    
    # TODO: implement your own loop subdivision here
    mesh_subdivided = subdivision_loop(mesh,1);       
    
    # print the new mesh information and save the mesh
    print(f'Subdivided Mesh Info: {mesh_subdivided}')
    mesh_subdivided.export('bunny_subdivided_1.obj')

    start_time = time.time()

    # quadratic error mesh decimation
    ###mesh_decimated = mesh.simplify_quadric_decimation(4)
    
    # TODO: implement your own quadratic error mesh decimation here
    mesh_decimated = simplify_quadric_error(mesh, face_count=2000)
    
    # print the new mesh information and save the mesh
    print(f'Decimated Mesh Info: {mesh_decimated}')
    ### mesh_decimated.export('assets/assignment1/cube_decimated.obj')
    mesh_decimated.export('bunny_decimated_2000.obj')

    end_time = time.time()
    print(f"The code ran for {end_time - start_time} seconds.")