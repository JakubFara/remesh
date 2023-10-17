import meshio
import numpy as np
import warnings
from dolfin import MPI


class MeshLoader():
    def __init__(self, filename: str, cell_type: str = 'triangle', comm=None):
        """
        This class is designed to create dolfin:Mesh from .msh mesh.
        Arguments:
            filename: str: Name of the file. Has to be in format .msh.
            cell_tyle: str: It works only for triangle mesh.
        """

        if comm is None:
            self.comm = MPI.comm_world
        else:
            self.comm = comm
        self.filename = filename
        self.cell_type = cell_type
        if cell_type == 'triangle':
            self.dim = 2
        else:
            self.dim = 3
        self.msh = meshio.read(filename)
        # get data of meshes
        self.data = self.get_data()
        self.cells = self.get_cells()
        # load xdmf meshes
        self.dolfin_mesh = self._load_dolfin_mesh()
        if self.dim == 3:
            self.dolfin_facet_mesh = self._load_facet_dolfin_mesh()
        self.dolfin_line_mesh = self._load_line_dolfin_mesh()
        self.dolfin_vertex_mesh = self._load_vertex_dolfin_mesh()

    def _load_dolfin_mesh(self):
        if len(self.data[self.cell_type]) != 0:
            mesh = meshio.Mesh(
                points=self.msh.points[:, :self.dim],
                cells={self.cell_type: self.cells[self.cell_type]},
                cell_data={
                    "name_to_read": [self.data[self.cell_type]],
                }
            )
        else:
            mesh = meshio.Mesh(
                points=self.msh.points[:, :self.dim],
                cells={self.cell_type: self.cell_type},
            )

        meshio.write("cache/mesh.xdmf", mesh)
        from dolfin import XDMFFile, Mesh
        triangle_dolfin_mesh = Mesh(self.comm)
        with XDMFFile(self.comm, "cache/mesh.xdmf") as infile:
            infile.read(triangle_dolfin_mesh)
        return triangle_dolfin_mesh

    def _load_facet_dolfin_mesh(self):
        if self.dim == 2:
            return None
        if len(self.cells["triangle"]) != 0:
            facet_mesh = meshio.Mesh(
                points=self.msh.points[:, :self.dim],
                cells={'triangle': self.cells["triangle"]},
                cell_data={
                    "name_to_read": [self.data["triangle"]],
                }
            )
            from dolfin import XDMFFile, Mesh
            meshio.write("cache/facet_mesh.xdmf", facet_mesh)
            facet_dolfin_mesh = Mesh(self.comm)
            with XDMFFile(self.comm, "cache/facet_mesh.xdmf") as infile:
                infile.read(facet_dolfin_mesh)
            return facet_dolfin_mesh
        return None

    def _load_line_dolfin_mesh(self):
        if len(self.cells["line"]) != 0:
            line_mesh = meshio.Mesh(
                points=self.msh.points[:, :self.dim],
                cells={'line': self.cells["line"]},
                cell_data={
                    "name_to_read": [self.data["line"]],
                }
            )
            from dolfin import XDMFFile, Mesh
            meshio.write("cache/line_mesh.xdmf", line_mesh)
            line_dolfin_mesh = Mesh(self.comm)
            with XDMFFile(self.comm, "cache/line_mesh.xdmf") as infile:
                infile.read(line_dolfin_mesh)
            return line_dolfin_mesh
        return None

    def _load_vertex_dolfin_mesh(self):
        if len(self.cells["vertex"]) != 0:
            vertex_mesh = meshio.Mesh(
                points=self.msh.points[:, :self.dim],
                cells={'vertex': self.vertex_cells},
                cell_data={
                    "name_to_read": [self.data["vertex"]],
                }
            )
            from dolfin import XDMFFile, Mesh
            meshio.write("cache/vertex_mesh.xdmf", vertex_mesh)
            vertex_dolfin_mesh = Mesh(self.comm)
            with XDMFFile("cache/line_mesh.xdmf") as infile:
                infile.read(vertex_dolfin_mesh)
            return vertex_dolfin_mesh
        return None

    def get_mesh_function(self, dim: int):
        if dim == self.dim:
            return self.cell_label()
        if dim == 1:
            return self.facet_label()

    def cell_label(self):
        from dolfin import XDMFFile, MeshValueCollection, cpp
        mvc = MeshValueCollection("size_t", self.dolfin_mesh, self.dim)
        # meshio.write("cache/mesh.xdmf", self.xdmf_mesh)
        with XDMFFile(self.comm, "cache/mesh.xdmf") as infile:
            infile.read(mvc, "name_to_read")
        mf = cpp.mesh.MeshFunctionSizet(self.dolfin_mesh, mvc)
        return mf

    def facet_label(self):
        if self.dim == 3:
           cache_file = "cache/facet_mesh.xdmf"
           exists = self.dolfin_facet_mesh
        elif self.dim == 2:
           cache_file = "cache/line_mesh.xdmf"
           exists = self.dolfin_line_mesh
        if exists:
            from dolfin import XDMFFile, MeshValueCollection, cpp
            mvc = MeshValueCollection("size_t", self.dolfin_mesh, self.dim - 1)
            # meshio.write("cache/line_mesh.xdmf", self.xdmf_line_mesh)
            with XDMFFile(self.comm, cache_file) as infile:
                infile.read(mvc, "name_to_read")
            mf = cpp.mesh.MeshFunctionSizet(self.dolfin_mesh, mvc)
            return mf
        else:
            warnings.warn('No facet markers!')
            return None

    def vertex_label(self):
        if self.dolfin_line_mesh:
            from dolfin import XDMFFile, MeshValueCollection, cpp
            mvc = MeshValueCollection("size_t", self.dolfin_mesh, 0)
            # meshio.write("cache/line_mesh.xdmf", self.xdmf_line_mesh)
            with XDMFFile(self.comm, "cache/vertex_mesh.xdmf") as infile:
                infile.read(mvc, "name_to_read")
            mf = cpp.mesh.MeshFunctionSizet(self.dolfin_mesh, mvc)
            return mf
        else:
            warnings.warn('No facet markers!')
            return None

    def get_data(self):
        data = {"vertex": [], "line": [], "triangle": [], "tetra": []}
        cell_data_dict = self.msh.cell_data_dict["gmsh:physical"]
        for key in cell_data_dict.keys():
            for cell_type in data.keys():
                if key == cell_type:
                    if len(data[cell_type]) == 0:
                        data[cell_type] = cell_data_dict[cell_type]
                    else:
                        data[cell_type] = np.vstack(
                            [
                                data[cell_type],
                                cell_data_dict[key]
                            ]
                        )
        return data

    def get_cells(self):
        cells = {"vertex": [], "line": [], "triangle": [], "tetra": []}
        for cell in self.msh.cells:
            for cell_type in cells.keys():
                if cell.type == cell_type:
                    if len(cells[cell_type]) == 0:
                        cells[cell_type] = cell.data
                    else:
                        cells[cell_type] = np.vstack([cells[cell_type], cell.data])
        return cells

    def save_data(self, name: str = 'mesh.h5'):
        facet_marker = self.facet_label()
        cell_marker = self.cell_label()
        from dolfin import HDF5File, MPI
        with HDF5File(self.comm, name, 'w') as hfile:
            hfile.write(self.dolfin_mesh, '/mesh')
            hfile.write(cell_marker, '/cell_marker')
            if facet_marker:
                hfile.write(facet_marker, '/facet_marker')
