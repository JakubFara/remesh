from dolfin import UnitSquareMesh, MeshFunction, facets, MPI, HDF5File, Mesh
from src import remesh
from src.generate_mesh import MeshLoader

comm = MPI.comm_world
comm_self = MPI.comm_self


mesh = UnitSquareMesh(10, 10)
edge_marker = MeshFunction('size_t', mesh, 1)
for facet in facets(mesh):
    x = facet.midpoint().x()
    y = facet.midpoint().y()
    if x == 0:
        edge_marker[facet] = 1
    elif x == 1:
        edge_marker[facet] = 2
    elif y == 0:
        edge_marker[facet] = 3
    elif y == 1:
        edge_marker[facet] = 4


remesh(mesh, 'data/mesh.h5', edge_marker=edge_marker,  comm=None, comm_self=None)


with HDF5File(comm, "data/mesh.h5", "r") as hdf_file:
    mesh = Mesh(comm)
    hdf_file.read(mesh, "/mesh", False)
    edge_marker = MeshFunction('size_t', mesh, 1)
    hdf_file.read(edge_marker, "/facet_marker")
