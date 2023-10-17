from src.generate_mesh import MeshLoader
from dolfin import MeshFunction, facets, MPI, HDF5File, Mesh
import dolfin as df
import gmsh


def find_initial_edge(mesh: df.Mesh):
    for f in df.facets(mesh):
        if len([c for c in df.cells(f)]) == 1:
            return f


def boundary(mesh: df.Mesh, mesh_function: df.MeshFunction = None):
    """
    find boundary vertices and return them sorted
    """
    initial_edge = find_initial_edge(mesh)
    initial_index = initial_edge.index()
    for initial_vertex in df.vertices(initial_edge):
        vertex = initial_vertex
    previous_index = initial_edge.index()
    edge = initial_edge
    vertices = []
    edge_counter = 0
    marks = {}
    while True:
        for vertex_it in df.vertices(edge):
            if vertex_it.index() != vertex.index():
                vertices.append([
                    vertex_it.midpoint().x(), vertex_it.midpoint().y()]
                )
                vertex = vertex_it
                break
        for edge_it in df.facets(vertex):
            if (
                edge_it.index() != previous_index and
                len([c for c in df.cells(edge_it)]) == 1
            ):
                # use mesh_function
                # matker = mesh_function[edge]
                # "append dict[marker]"
                if mesh_function is not None:
                    if marks.get(mesh_function[edge_it]):
                        marks[mesh_function[edge_it]].append(edge_counter)
                    else:
                        marks[mesh_function[edge_it]] = [edge_counter]
                    edge_counter += 1
                previous_index = edge_it.index()
                edge = edge_it
                break
        if edge.index() == initial_index:
            return vertices, marks


def create_gmsh_from_edges(vertices, fine_const, filename, marks,
                           vizual=False):
    # Initialize gmsh:
    # gmsh.initialize()
    points = []
    for vertex in vertices:
        points.append(
            gmsh.model.geo.add_point(vertex[0], vertex[1], 0, fine_const)
        )
    edges = []
    point0 = points[0]
    for point in points[1:]:
        edges.append(gmsh.model.geo.add_line(point0, point))
        point0 = point
    edges.append(gmsh.model.geo.add_line(point0, points[0]))
    faces = gmsh.model.geo.addCurveLoop(edges)
    gmsh.model.geo.addPlaneSurface([faces], 1)
    gmsh.model.geo.synchronize()
    gmsh.model.addPhysicalGroup(2, [1], 1)
    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(2)
    for mark, indices in marks.items():
        gmsh.model.addPhysicalGroup(
            1, [edges[index] for index in indices], mark
        )
    gmsh.write(filename)
    # Creates  graphical user interface
    # if 'close' not in sys.argv:
    if vizual:
        gmsh.fltk.run()

    # It finalize the Gmsh API
    gmsh.finalize()


def recreate_mesh(mesh: df.Mesh, fine_const, filename, mesh_function,
                  vizual=False):
    boundary_vertices, marks = boundary(mesh, mesh_function=mesh_function)
    create_gmsh_from_edges(
        boundary_vertices, fine_const, filename, marks, vizual=vizual,
    )


def remesh(mesh, mesh_file, edge_marker=None, comm=None, comm_self=None):
    if comm is None:
        comm = MPI.comm_world
    if comm_self is None:
        comm_self = MPI.comm_self
    with HDF5File(comm, "cache/old_mesh.h5", "w") as hdf_file:
        hdf_file.write(mesh, "/mesh")
        if edge_marker is not None:
            hdf_file.write(edge_marker, "/edge_marker")

    gmsh.initialize()
    if comm.Get_rank() == 0:
        local_mesh = Mesh(comm_self)
        with HDF5File(comm_self, "cache/old_mesh.h5", "r") as hdf_file:
            hdf_file.read(local_mesh, "/mesh", False)
            local_edge_marker = MeshFunction('size_t', local_mesh, 1)
            if edge_marker is not None:
                hdf_file.read(local_edge_marker, "/edge_marker")
            else:
                local_edge_marker = None

        filename = 'cache/mesh.msh'
        recreate_mesh(
            local_mesh, 0.01, filename, local_edge_marker,
            vizual=False
        )
        mesh_loader = MeshLoader(filename, 'triangle', comm=comm_self)
        mesh_loader.save_data(name=mesh_file)
