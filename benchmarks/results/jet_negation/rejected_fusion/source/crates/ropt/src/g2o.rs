//! Strict 2D g2o reader. Unsupported records are errors, never silently dropped.
use crate::factor::{FactorGraph, FactorOptions, VariableId, VariableKind, read_se2, se2_values};
use crate::manifold::{SE2, SO2};
use crate::noise::NoiseModel;
use crate::pose_graph::{add_relative_pose2_factor, add_relative_pose2_factor_analytic};
use crate::{DMatrix, DVector, Error, Loss};
use nalgebra::Vector2;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::io::{self, BufRead, Write};

#[derive(Debug)]
pub enum G2oError {
    Io(io::Error),
    /// One-based input line. Zero indicates a whole-graph validation failure.
    Invalid {
        line: usize,
        message: String,
    },
    Optimization(Error),
}
impl fmt::Display for G2oError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "g2o I/O: {e}"),
            Self::Invalid { line, message } => write!(f, "g2o line {line}: {message}"),
            Self::Optimization(e) => write!(f, "g2o graph: {e}"),
        }
    }
}
impl std::error::Error for G2oError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            Self::Optimization(e) => Some(e),
            _ => None,
        }
    }
}
impl From<io::Error> for G2oError {
    fn from(e: io::Error) -> Self {
        Self::Io(e)
    }
}
impl From<Error> for G2oError {
    fn from(e: Error) -> Self {
        Self::Optimization(e)
    }
}
fn invalid(line: usize, message: impl Into<String>) -> G2oError {
    G2oError::Invalid {
        line,
        message: message.into(),
    }
}
fn id(s: &str, line: usize) -> Result<u64, G2oError> {
    s.parse()
        .map_err(|_| invalid(line, "expected nonnegative integer ID"))
}
fn number(s: &str, line: usize) -> Result<f64, G2oError> {
    let v: f64 = s
        .parse()
        .map_err(|_| invalid(line, "expected floating point number"))?;
    if !v.is_finite() {
        return Err(invalid(line, "nonfinite number"));
    }
    Ok(v)
}
fn pose(v: &[&str], line: usize) -> Result<SE2, G2oError> {
    Ok(SE2 {
        translation: Vector2::new(number(v[0], line)?, number(v[1], line)?),
        rotation: SO2::exp(number(v[2], line)?),
    })
}
#[derive(Debug, Clone)]
pub struct Edge2 {
    pub from: u64,
    pub to: u64,
    pub measurement: SE2,
    pub information: DMatrix<f64>,
    line: usize,
}
/// Parsed data is read-only; graph construction validates references and anchors.
#[derive(Debug, Clone, Default)]
pub struct PoseGraph2 {
    vertices: BTreeMap<u64, SE2>,
    edges: Vec<Edge2>,
    fixed: BTreeSet<u64>,
}
#[derive(Debug, Clone, Copy, Default)]
pub enum PoseJacobian {
    #[default]
    AutoDiff,
    Analytic,
}
impl PoseGraph2 {
    pub fn vertices(&self) -> &BTreeMap<u64, SE2> {
        &self.vertices
    }
    pub fn edges(&self) -> &[Edge2] {
        &self.edges
    }
    pub fn fixed(&self) -> &BTreeSet<u64> {
        &self.fixed
    }
    /// Accept VERTEX_SE2, EDGE_SE2, FIX, blank lines and # comments.
    /// VERTEX2/EDGE2 aliases use the same g2o upper-triangle information order;
    /// TORO's different matrix order is not supported or guessed.
    pub fn from_reader(reader: impl BufRead) -> Result<Self, G2oError> {
        let mut graph = Self::default();
        let mut fixed_lines = BTreeMap::new();
        for (index, line) in reader.lines().enumerate() {
            let line = line?;
            let line_number = index + 1;
            let text = line.split('#').next().unwrap_or("");
            let v: Vec<_> = text.split_whitespace().collect();
            if v.is_empty() {
                continue;
            }
            let fields = match v[0] {
                "VERTEX_SE2" | "VERTEX2" => 5,
                "EDGE_SE2" | "EDGE2" => 12,
                "FIX" => 2,
                tag => return Err(invalid(line_number, format!("unsupported record {tag}"))),
            };
            if v.len() != fields {
                return Err(invalid(
                    line_number,
                    format!("{} expects {} fields", v[0], fields - 1),
                ));
            }
            match v[0] {
                "VERTEX_SE2" | "VERTEX2" => {
                    let key = id(v[1], line_number)?;
                    let value = pose(&v[2..5], line_number)?;
                    if graph.vertices.insert(key, value).is_some() {
                        return Err(invalid(line_number, "duplicate vertex ID"));
                    }
                }
                "EDGE_SE2" | "EDGE2" => {
                    let from = id(v[1], line_number)?;
                    let to = id(v[2], line_number)?;
                    if from == to {
                        return Err(invalid(line_number, "self edge is unsupported"));
                    }
                    let m = pose(&v[3..6], line_number)?;
                    let w = v[6..]
                        .iter()
                        .map(|s| number(s, line_number))
                        .collect::<Result<Vec<_>, _>>()?;
                    let information = DMatrix::from_row_slice(
                        3,
                        3,
                        &[w[0], w[1], w[2], w[1], w[3], w[4], w[2], w[4], w[5]],
                    );
                    NoiseModel::from_information(information.clone()).map_err(|e| {
                        invalid(line_number, format!("invalid SPD information: {e}"))
                    })?;
                    graph.edges.push(Edge2 {
                        from,
                        to,
                        measurement: m,
                        information,
                        line: line_number,
                    });
                }
                "FIX" => {
                    let key = id(v[1], line_number)?;
                    graph.fixed.insert(key);
                    fixed_lines.insert(key, line_number);
                }
                _ => unreachable!(),
            }
        }
        if graph.vertices.is_empty() {
            return Err(invalid(0, "no vertices"));
        }
        for edge in &graph.edges {
            if !graph.vertices.contains_key(&edge.from) || !graph.vertices.contains_key(&edge.to) {
                return Err(invalid(edge.line, "edge references unknown vertex"));
            }
        }
        for (key, line) in fixed_lines {
            if !graph.vertices.contains_key(&key) {
                return Err(invalid(line, "FIX references unknown vertex"));
            }
        }
        Ok(graph)
    }
    /// Honor all FIX records; otherwise anchor the smallest ID in each connected component.
    /// Isolated vertices are rejected, since they have no measurement constraints.
    /// Some(loss) applies jointly to the whitened three-component edge residual.
    pub fn build(&self, loss: Option<Loss>) -> Result<LoadedPoseGraph2, G2oError> {
        self.build_with_jacobians(loss, PoseJacobian::AutoDiff)
    }
    /// Same graph/anchors/objective with an explicit derivative implementation.
    pub fn build_with_jacobians(
        &self,
        loss: Option<Loss>,
        jacobians: PoseJacobian,
    ) -> Result<LoadedPoseGraph2, G2oError> {
        if self.vertices.is_empty() {
            return Err(invalid(0, "no vertices"));
        }
        let mut neighbors: BTreeMap<u64, Vec<u64>> =
            self.vertices.keys().map(|&id| (id, Vec::new())).collect();
        for e in &self.edges {
            neighbors.get_mut(&e.from).unwrap().push(e.to);
            neighbors.get_mut(&e.to).unwrap().push(e.from);
        }
        let mut anchors = self.fixed.clone();
        let mut visited = BTreeSet::new();
        for (&key, adjacent) in &neighbors {
            if adjacent.is_empty() {
                return Err(invalid(0, format!("isolated vertex {key}")));
            }
            if visited.contains(&key) {
                continue;
            }
            let mut stack = vec![key];
            let mut anchored = false;
            while let Some(next) = stack.pop() {
                if !visited.insert(next) {
                    continue;
                }
                anchored |= self.fixed.contains(&next);
                stack.extend(
                    neighbors[&next]
                        .iter()
                        .copied()
                        .filter(|n| !visited.contains(n)),
                );
            }
            if !anchored {
                anchors.insert(key);
            }
        }
        let mut graph = FactorGraph::new();
        let mut variables = BTreeMap::new();
        for (&key, &value) in &self.vertices {
            let variable = graph.add_variable(VariableKind::SE2, se2_values(value))?;
            if anchors.contains(&key) {
                graph.set_fixed(variable, true)?;
            }
            variables.insert(key, variable);
        }
        for e in &self.edges {
            let register = match jacobians {
                PoseJacobian::AutoDiff => add_relative_pose2_factor,
                PoseJacobian::Analytic => add_relative_pose2_factor_analytic,
            };
            register(
                &mut graph,
                variables[&e.from],
                variables[&e.to],
                e.measurement,
                FactorOptions {
                    noise: NoiseModel::from_information(e.information.clone())?,
                    loss,
                },
            )?;
        }
        Ok(LoadedPoseGraph2 {
            graph,
            variables,
            anchors,
        })
    }
    /// Write modern g2o tags, preserving IDs, edges, full information and original FIX records.
    /// Added gauge anchors are a solver policy and are not added to exported data.
    pub fn write(
        &self,
        writer: &mut impl Write,
        poses: &BTreeMap<u64, SE2>,
    ) -> Result<(), G2oError> {
        if poses.len() != self.vertices.len() || poses.keys().ne(self.vertices.keys()) {
            return Err(invalid(0, "output pose IDs differ from input"));
        }
        for (&id, &p) in poses {
            VariableKind::SE2.validate(&se2_values(p))?;
            writeln!(
                writer,
                "VERTEX_SE2 {id} {:.17e} {:.17e} {:.17e}",
                p.translation.x,
                p.translation.y,
                p.rotation.log()
            )?;
        }
        for e in &self.edges {
            let m = e.measurement;
            let w = &e.information;
            writeln!(
                writer,
                "EDGE_SE2 {} {} {:.17e} {:.17e} {:.17e} {:.17e} {:.17e} {:.17e} {:.17e} {:.17e} {:.17e}",
                e.from,
                e.to,
                m.translation.x,
                m.translation.y,
                m.rotation.log(),
                w[(0, 0)],
                w[(0, 1)],
                w[(0, 2)],
                w[(1, 1)],
                w[(1, 2)],
                w[(2, 2)]
            )?;
        }
        for id in &self.fixed {
            writeln!(writer, "FIX {id}")?;
        }
        Ok(())
    }
}
pub struct LoadedPoseGraph2 {
    pub graph: FactorGraph,
    pub variables: BTreeMap<u64, VariableId>,
    pub anchors: BTreeSet<u64>,
}
impl LoadedPoseGraph2 {
    pub fn poses(&self, x: &DVector<f64>) -> Result<BTreeMap<u64, SE2>, Error> {
        self.variables
            .iter()
            .map(|(&key, &id)| Ok((key, read_se2(&self.graph.value(x, id)?)?)))
            .collect()
    }
}
