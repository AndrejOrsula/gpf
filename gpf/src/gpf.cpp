/// Grometric Primitive Fitting

////////////////////
/// DEPENDENCIES ///
////////////////////

// ROS 2
#include <rclcpp/rclcpp.hpp>
#include <pcl_conversions/pcl_conversions.h>

// ROS 2 interfaces
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <geometric_primitive_msgs/msg/geometric_primitive_list_stamped.hpp>

// PCL
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/segmentation/supervoxel_clustering.h>
#include <pcl/segmentation/cpc_segmentation.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/sample_consensus/sac_model_sphere.h>
#include <pcl/sample_consensus/sac_model_cylinder.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/point_cloud_color_handlers.h>

///////////////
/// DEFINES ///
///////////////

/// If defined, CPC is used for segmentation instead of LCCP
#define USE_CPC
/// Define what sample consensus estimator to utilise by changing `SAMPLE_CONSENSUS_ESTIMATOR`
#define SAMPLE_CONSENSUS_ESTIMATOR_RANSAC 0
#define SAMPLE_CONSENSUS_ESTIMATOR_LMEDS 1
#define SAMPLE_CONSENSUS_ESTIMATOR_MSAC 2
#define SAMPLE_CONSENSUS_ESTIMATOR_RRANSAC 3
#define SAMPLE_CONSENSUS_ESTIMATOR_RMSAC 4
#define SAMPLE_CONSENSUS_ESTIMATOR_MLESAC 5
#define SAMPLE_CONSENSUS_ESTIMATOR_PROSAC 6
#define SAMPLE_CONSENSUS_ESTIMATOR SAMPLE_CONSENSUS_ESTIMATOR_RANSAC

/////////////////
/// CONSTANTS ///
/////////////////

/// The name of this node
const std::string NODE_NAME = "gpf";
/// Determines whether to update parameters during run-time on each message callback
const bool UPDATE_PARAMETERS_CONTINUOUSLY = true;

/////////////
/// TYPES ///
/////////////

/// Type of the input point cloud contents
typedef pcl::PointXYZRGB PointT;

/// Type of the sample consensus estimator to use
#if SAMPLE_CONSENSUS_ESTIMATOR == SAMPLE_CONSENSUS_ESTIMATOR_RANSAC
#include <pcl/sample_consensus/ransac.h>
typedef pcl::RandomSampleConsensus<PointT> SampleConsensusEstimator;
#elif SAMPLE_CONSENSUS_ESTIMATOR == SAMPLE_CONSENSUS_ESTIMATOR_LMEDS
#include <pcl/sample_consensus/lmeds.h>
typedef pcl::LeastMedianSquares<PointT> SampleConsensusEstimator;
#elif SAMPLE_CONSENSUS_ESTIMATOR == SAMPLE_CONSENSUS_ESTIMATOR_MSAC
#include <pcl/sample_consensus/msac.h>
typedef pcl::MEstimatorSampleConsensus<PointT> SampleConsensusEstimator;
#elif SAMPLE_CONSENSUS_ESTIMATOR == SAMPLE_CONSENSUS_ESTIMATOR_RRANSAC
#include <pcl/sample_consensus/rransac.h>
typedef pcl::RandomizedRandomSampleConsensus<PointT> SampleConsensusEstimator;
#elif SAMPLE_CONSENSUS_ESTIMATOR == SAMPLE_CONSENSUS_ESTIMATOR_RMSAC
#include <pcl/sample_consensus/rmsac.h>
typedef pcl::RandomizedMEstimatorSampleConsensus<PointT> SampleConsensusEstimator;
#elif SAMPLE_CONSENSUS_ESTIMATOR == SAMPLE_CONSENSUS_ESTIMATOR_MLESAC
#include <pcl/sample_consensus/mlesac.h>
typedef pcl::MaximumLikelihoodSampleConsensus<PointT> SampleConsensusEstimator;
#elif SAMPLE_CONSENSUS_ESTIMATOR == SAMPLE_CONSENSUS_ESTIMATOR_PROSAC
#include <pcl/sample_consensus/prosac.h>
typedef pcl::ProgressiveSampleConsensus<PointT> SampleConsensusEstimator;
#endif

//////////////////////////////////
/// HELPER STRUCTS AND CLASSES ///
//////////////////////////////////

class Shape
{
public:
  uint32_t id;
  Eigen::VectorXf _raw_coefficients;
  std::vector<int> inliers;
  struct
  {
    float inlier_proportion;
  } validity;
};

class Plane : public Shape
{
public:
  Eigen::Hyperplane<float, 3> model;

  void init(const uint32_t segment_id)
  {
    id = segment_id;
    model = Eigen::Hyperplane<float, 3>(_raw_coefficients.head<3>(), _raw_coefficients[3]);
  }
};

class Sphere : public Shape
{
public:
  struct
  {
    Eigen::Vector3f centre;
    float radius;
  } model;

  void init(const uint32_t segment_id)
  {
    id = segment_id;
    model.centre = _raw_coefficients.head<3>();
    model.radius = _raw_coefficients[3];
  }
};

class Cylinder : public Shape
{
public:
  struct
  {
    struct
    {
      Eigen::Vector3f position;
      Eigen::Quaternion<double> orientation;
    } pose;
    float height;
    float radius;
  } model;
  struct
  {
    Eigen::ParametrizedLine<float, 3> axis;
    float radius;
  } pcl_model;

  void init(const uint32_t segment_id)
  {
    id = segment_id;
    pcl_model.axis = Eigen::ParametrizedLine<float, 3>(_raw_coefficients.head<3>(), _raw_coefficients.head<6>().tail<3>());
    model.radius = pcl_model.radius = _raw_coefficients[6];
  };
};

struct ParametersSampleConsensusCommon
{
  bool enable;
  float distance_threshold;
  float probability;
  uint16_t max_iterations;
  float min_inlier_proportion;
};

struct ParametersSampleConsensusPlane : ParametersSampleConsensusCommon
{
  float merge_precision;
};

struct ParametersSampleConsensusWithRadius : ParametersSampleConsensusCommon
{
  float min_radius;
  float max_radius;
};

struct Parameters
{
  // Input preprocessing
  bool downsample_input;
  float voxel_leaf_size_xy;
  float voxel_leaf_size_z;

  // Supervoxels
  bool use_single_camera_transform;
  float voxel_resolution;
  float seed_resolution;
  float spatial_importance;
  float color_importance;
  bool enable_normals;
  float normal_search_radius;
  float normal_importance;
  bool enable_supervoxel_refinement;
  uint8_t supervoxel_refinement_iterations;

  // LCCP Segmentation
  float concavity_tolerance_threshold;
  float smoothness_threshold;
  uint16_t min_segment_size;
  uint8_t k_factor;
  bool sanity_criterion;

  // CPC Segmentation
  uint8_t max_cuts;
  uint16_t cutting_min_segments;
  float cutting_min_score;
  bool local_constrain;
  bool directed_cutting;
  bool clean_cutting;
  uint16_t ransac_iterations;

  // Sample Consensus
  uint16_t sac_min_segment_size;
  ParametersSampleConsensusPlane plane;
  ParametersSampleConsensusWithRadius sphere;
  ParametersSampleConsensusWithRadius cylinder;

  // Miscellaneous
  bool publish_markers;
  bool visualise;
};

//////////////////
/// NODE CLASS ///
//////////////////

/// Class representation of this node
class GeometricPrimitiveFitting : public rclcpp::Node
{
public:
  /// Constructor
  GeometricPrimitiveFitting();

private:
  /// Subscriber to the point cloud
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_point_cloud_;

  /// Publisher of primitives
  rclcpp::Publisher<geometric_primitive_msgs::msg::GeometricPrimitiveListStamped>::SharedPtr pub_primitives_;
  /// Publisher of visualisation markers
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_markers_;

  /// PLC visualised
  pcl::visualization::PCLVisualizer::Ptr viewer_;

  /// List of all parameters of this node
  Parameters parameters_;

  /// Callback called each time a message is received on all topics
  void point_cloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg_point_cloud);
  /// Declaration of all node parameters
  void declare_parameters();
  /// Update of all node parameters
  void update_parameters();
};

GeometricPrimitiveFitting::GeometricPrimitiveFitting() : Node(NODE_NAME)
{
  // Synchronize the subscriptions under a single callback
  rclcpp::QoS qos_pc = rclcpp::QoS(rclcpp::QoSInitialization::from_rmw(rmw_qos_profile_default));
  sub_point_cloud_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      "camera/pointcloud", qos_pc, std::bind(&GeometricPrimitiveFitting::point_cloud_callback, this, std::placeholders::_1));

  // Declare parameters of the node
  declare_parameters();

  // Register publisher of geometric primitives
  rclcpp::QoS qos_primitives = rclcpp::QoS(rclcpp::QoSInitialization::from_rmw(rmw_qos_profile_default));
  pub_primitives_ = this->create_publisher<geometric_primitive_msgs::msg::GeometricPrimitiveListStamped>("geometric_primitives", qos_primitives);

  // Register publisher of visualisation markers
  if (parameters_.publish_markers)
  {
    rclcpp::QoS qos_markers = rclcpp::QoS(rclcpp::QoSInitialization::from_rmw(rmw_qos_profile_default));
    pub_markers_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("visualisation_markers", qos_markers);
  }

  // Setup viewer config
  if (parameters_.visualise)
  {
    viewer_.reset(new pcl::visualization::PCLVisualizer("GPF Viewer"));
    viewer_->setBackgroundColor(48 / 255.0, 48 / 255.0, 48 / 255.0);
    viewer_->initCameraParameters();
  }

  RCLCPP_INFO(this->get_logger(), "Node initialised");
}

void GeometricPrimitiveFitting::point_cloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg_point_cloud)
{
  RCLCPP_DEBUG(this->get_logger(), "Received message for processing");

  // Convert msg to PCL
  pcl::PointCloud<PointT>::Ptr point_cloud(new pcl::PointCloud<PointT>);
  pcl::fromROSMsg(*msg_point_cloud, *point_cloud);

  // Update parameters, if desired
  if (UPDATE_PARAMETERS_CONTINUOUSLY)
  {
    update_parameters();
  }

  //
  // Preprocessing
  //

  // Downsample input to speed up the computations, if desired
  if (parameters_.downsample_input)
  {
    pcl::VoxelGrid<PointT> voxel_grid;
    voxel_grid.setInputCloud(point_cloud);
    voxel_grid.setLeafSize(parameters_.voxel_leaf_size_xy,
                           parameters_.voxel_leaf_size_xy,
                           parameters_.voxel_leaf_size_z);
    voxel_grid.filter(*point_cloud);
  }

  //
  // Supervoxels
  //

  // Setup supervoxel clustering
  pcl::SupervoxelClustering<PointT> supervoxel_clustering(parameters_.voxel_resolution,
                                                          parameters_.seed_resolution);
  supervoxel_clustering.setInputCloud(point_cloud);
  supervoxel_clustering.setUseSingleCameraTransform(parameters_.use_single_camera_transform);
  supervoxel_clustering.setSpatialImportance(parameters_.spatial_importance);
  supervoxel_clustering.setColorImportance(parameters_.color_importance);

  // Estimate the normals and use them during clustering, if desired
  if (parameters_.enable_normals)
  {
    pcl::NormalEstimation<PointT, pcl::Normal> normal_estimator;
    normal_estimator.setInputCloud(point_cloud);
    pcl::search::KdTree<PointT>::Ptr tree_n(new pcl::search::KdTree<PointT>());
    normal_estimator.setSearchMethod(tree_n);
    normal_estimator.setRadiusSearch(parameters_.normal_search_radius);
    pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(new pcl::PointCloud<pcl::Normal>());
    normal_estimator.compute(*cloud_normals);

    supervoxel_clustering.setNormalCloud(cloud_normals);
  }
  supervoxel_clustering.setNormalImportance(parameters_.normal_importance);

  // Segment and extract the supervoxel clusters
  std::map<std::uint32_t, pcl::Supervoxel<PointT>::Ptr> supervoxel_clusters;
  supervoxel_clustering.extract(supervoxel_clusters);

  // Refine supervoxels, if desired
  if (parameters_.enable_supervoxel_refinement)
  {
    supervoxel_clustering.refineSupervoxels(parameters_.supervoxel_refinement_iterations, supervoxel_clusters);
  }

  // Get supervoxel adjacency
  std::multimap<std::uint32_t, std::uint32_t> supervoxel_adjacency;
  supervoxel_clustering.getSupervoxelAdjacency(supervoxel_adjacency);

  //
  // Segmentation
  //

  // Setup CPC with its LCCP preprocesing or just pure LCCP
#ifdef USE_CPC
  pcl::CPCSegmentation<PointT> segmentation;
#else
  pcl::LCCPSegmentation<PointT> segmentation;
#endif
  segmentation.setInputSupervoxels(supervoxel_clusters, supervoxel_adjacency);
  segmentation.setConcavityToleranceThreshold(parameters_.concavity_tolerance_threshold);
  segmentation.setSmoothnessCheck(true,
                                  parameters_.voxel_resolution,
                                  parameters_.seed_resolution,
                                  parameters_.smoothness_threshold);
  segmentation.setMinSegmentSize(parameters_.min_segment_size);
  segmentation.setKFactor(parameters_.k_factor);
  segmentation.setSanityCheck(parameters_.sanity_criterion);
#ifdef USE_CPC
  segmentation.setCutting(parameters_.max_cuts,
                          parameters_.cutting_min_segments,
                          parameters_.cutting_min_score,
                          parameters_.local_constrain,
                          parameters_.directed_cutting,
                          parameters_.clean_cutting);
  segmentation.setRANSACIterations(parameters_.ransac_iterations);
#endif

  // Segment
  segmentation.segment();

  // Determine labels for the segmented supervoxels
  std::map<std::uint32_t, std::set<std::uint32_t>> segmentation_map;
  segmentation.getSegmentToSupervoxelMap(segmentation_map);

  // Create separate point clouds for each segment
  std::map<std::uint32_t, pcl::PointCloud<PointT>::Ptr> segmented_point_clouds;
  for (const auto &[segment_id, voxel_ids] : segmentation_map)
  {
    // Fill each point cloud with all corresponding supervoxels
    pcl::PointCloud<PointT>::Ptr segmented_point_cloud(new pcl::PointCloud<PointT>);
    for (const auto &voxel_id : voxel_ids)
    {
      *segmented_point_cloud += *supervoxel_clusters.at(voxel_id)->voxels_;
    }
    segmented_point_clouds.insert(std::pair(segment_id, segmented_point_cloud));
  }

  //
  // Sample Consensus
  //

  // Create individual groups for the supported primitives
  std::vector<Plane> planes;
  std::vector<Sphere> spheres;
  std::vector<Cylinder> cylinders;

  // Iterate over all segmented point clouds
  for (const auto &[segment_id, segment_cloud] : segmented_point_clouds)
  {
    // Make sure the point cloud has enough points
    if (segment_cloud->size() < parameters_.sac_min_segment_size)
    {
      continue;
    }

    // Create nullptr of sample consensus implementation
    SampleConsensusEstimator::Ptr sac;

    // Create nullptr of sample consensus models
    pcl::SampleConsensusModelPlane<PointT>::Ptr sac_model_plane;
    pcl::SampleConsensusModelSphere<PointT>::Ptr sac_model_sphere;
    pcl::SampleConsensusModelCylinder<PointT, pcl::Normal>::Ptr sac_model_cylinder;

    // Create new consideration of the geometric primitives
    Plane plane;
    Sphere sphere;
    Cylinder cylinder;

    // Setup and compute sample consensus for plane model
    if (parameters_.plane.enable)
    {
      sac_model_plane.reset(new pcl::SampleConsensusModelPlane<PointT>(segment_cloud));

      sac.reset(new SampleConsensusEstimator(sac_model_plane));
      sac->setDistanceThreshold(parameters_.plane.distance_threshold);
      sac->setProbability(parameters_.plane.probability);
      sac->setMaxIterations(parameters_.plane.max_iterations);

      sac->computeModel();
      sac->getModelCoefficients(plane._raw_coefficients);

      sac->getInliers(plane.inliers);
      plane.validity.inlier_proportion = (float)plane.inliers.size() / (float)segment_cloud->size();
    }

    // Setup and compute sample consensus for sphere model
    if (parameters_.sphere.enable)
    {
      sac_model_sphere.reset(new pcl::SampleConsensusModelSphere<PointT>(segment_cloud));
      sac_model_sphere->setRadiusLimits(parameters_.sphere.min_radius, parameters_.sphere.max_radius);

      sac.reset(new SampleConsensusEstimator(sac_model_sphere));
      sac->setDistanceThreshold(parameters_.sphere.distance_threshold);
      sac->setProbability(parameters_.sphere.probability);
      sac->setMaxIterations(parameters_.sphere.max_iterations);

      sac->computeModel();
      sac->getModelCoefficients(sphere._raw_coefficients);

      sac->getInliers(sphere.inliers);
      sphere.validity.inlier_proportion = (float)sphere.inliers.size() / (float)segment_cloud->size();
    }

    // Setup and compute sample consensus for cylinder model
    if (parameters_.cylinder.enable)
    {
      // Estimate normals for the segmented point cloud - required for cylinder sac
      pcl::NormalEstimation<PointT, pcl::Normal> normal_estimator;
      normal_estimator.setInputCloud(segment_cloud);
      pcl::search::KdTree<PointT>::Ptr tree_n(new pcl::search::KdTree<PointT>());
      normal_estimator.setSearchMethod(tree_n);
      normal_estimator.setRadiusSearch(parameters_.normal_search_radius);
      pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(new pcl::PointCloud<pcl::Normal>());
      normal_estimator.compute(*cloud_normals);

      sac_model_cylinder.reset(new pcl::SampleConsensusModelCylinder<PointT, pcl::Normal>(segment_cloud));
      sac_model_cylinder->setInputNormals(cloud_normals);
      sac_model_cylinder->setRadiusLimits(parameters_.cylinder.min_radius, parameters_.cylinder.max_radius);

      sac.reset(new SampleConsensusEstimator(sac_model_cylinder));
      sac->setDistanceThreshold(parameters_.cylinder.distance_threshold);
      sac->setProbability(parameters_.cylinder.probability);
      sac->setMaxIterations(parameters_.cylinder.max_iterations);

      sac->computeModel();
      sac->getModelCoefficients(cylinder._raw_coefficients);

      sac->getInliers(cylinder.inliers);
      cylinder.validity.inlier_proportion = (float)cylinder.inliers.size() / (float)segment_cloud->size();
    }

    RCLCPP_DEBUG(this->get_logger(), "Inlier proportions; plane: %f\tsphere: %f\tcylinder: %f", plane.validity.inlier_proportion, sphere.validity.inlier_proportion, cylinder.validity.inlier_proportion);

    // Create a local copy of enable flags that get overwritten if number of inliers/variance is not satisfactory
    bool consider_plane = parameters_.plane.enable;
    bool consider_sphere = parameters_.sphere.enable;
    bool consider_cylinder = parameters_.cylinder.enable;

    // Try to select the model with the lowest variance that has satisfactory number of inliers and variance
    for (;;)
    {
      if (consider_plane && (plane.validity.inlier_proportion > sphere.validity.inlier_proportion && plane.validity.inlier_proportion > cylinder.validity.inlier_proportion))
      {
        // Make sure the inlier proportion is satisfactory
        if (plane.validity.inlier_proportion < parameters_.plane.min_inlier_proportion)
        {
          consider_plane = false;
          continue;
        }

        // Initialise plane from the coefficients and segment id
        plane.init(segment_id);

        // Push with the rest of the planes
        planes.push_back(plane);
      }
      else if (consider_sphere && (sphere.validity.inlier_proportion > cylinder.validity.inlier_proportion))
      {
        // Make sure the inlier proportion is satisfactory
        if (sphere.validity.inlier_proportion < parameters_.sphere.min_inlier_proportion)
        {
          consider_sphere = false;
          continue;
        }

        // Initialise sphere from the coefficients and segment id
        sphere.init(segment_id);

        // Push with the rest of the spheres
        spheres.push_back(sphere);
      }
      else if (consider_cylinder)
      {
        // Make sure the inlier proportion is satisfactory
        if (cylinder.validity.inlier_proportion < parameters_.cylinder.min_inlier_proportion)
        {
          consider_cylinder = false;
          continue;
        }

        // Initialise cylinder from the coefficients and segment id
        cylinder.init(segment_id);

        // Extract cylinder limits from the associated pointcloud inliers
        // First create a hyperplane that intersects the cylinder, while being parallel with its flat surfaces
        Eigen::Hyperplane<float, 3> cylinder_plane(cylinder.pcl_model.axis.direction(), cylinder.pcl_model.axis.origin());

        // Iterate over all point cloud inliers and find the limits
        float min = 0.0, max = 0.0;
        for (auto &inlier_index : cylinder.inliers)
        {
          // Get signed distance to the point from hyperplane
          PointT point = segment_cloud->at(inlier_index);
          Eigen::Vector3f point_position(point.x, point.y, point.z);
          float signed_distance = cylinder_plane.signedDistance(point_position);

          // Overwrite the limits if new are found
          if (signed_distance < min)
          {
            min = signed_distance;
          }
          else if (signed_distance > max)
          {
            max = signed_distance;
          }
        }

        // Determine height of the cylinder
        cylinder.model.height = max - min;

        // Get centre of the cylinder and define it as the position
        cylinder.model.pose.position = (cylinder.pcl_model.axis.pointAt(min) + cylinder.pcl_model.axis.pointAt(max)) / 2.0;

        // Determne the orientation
        cylinder.model.pose.orientation.setFromTwoVectors(Eigen::Vector3d::UnitZ(), cylinder.pcl_model.axis.direction().cast<double>());

        // Push with the rest of the cylinders
        cylinders.push_back(cylinder);
      }

      // Break occurs only if the segment was pushed, or none is satisfactory
      break;
    }
  }

  ///
  /// Postprocessing
  ///

  // Merge approximately equal planes
  if (parameters_.plane.enable)
  {
    for (auto it = planes.begin(); it != planes.end(); it++)
    {
      for (auto jt = it + 1; jt != planes.end(); jt++)
      {
        if (it->model.isApprox(jt->model, parameters_.plane.merge_precision))
        {
          it->model.coeffs() = (it->model.coeffs() + jt->model.coeffs()) / 2.0;
          it->_raw_coefficients = it->model.coeffs();
          RCLCPP_DEBUG(this->get_logger(), "Merged plane #%d with pland #%d", it->id, jt->id);
          planes.erase(jt--);
        }
      }
    }
  }

  ///
  /// Create and publish msg
  ///

  geometric_primitive_msgs::msg::GeometricPrimitiveListStamped msg_primitives;
  msg_primitives.header = msg_point_cloud->header;

  // Planes
  for (auto &plane : planes)
  {
    geometric_primitive_msgs::msg::Plane p;
    p.id = plane.id;
    p.coefficients[p.COEFFICIENT_A] = plane.model.coeffs()[0];
    p.coefficients[p.COEFFICIENT_D] = plane.model.coeffs()[1];
    p.coefficients[p.COEFFICIENT_C] = plane.model.coeffs()[2];
    p.coefficients[p.COEFFICIENT_D] = plane.model.coeffs()[3];
    msg_primitives.primitives.planes.push_back(p);
  }

  // Spheres
  for (auto &sphere : spheres)
  {
    geometric_primitive_msgs::msg::Sphere s;
    s.id = sphere.id;
    s.centre.x = sphere.model.centre.x();
    s.centre.y = sphere.model.centre.y();
    s.centre.z = sphere.model.centre.z();
    s.radius = sphere.model.radius;
    msg_primitives.primitives.spheres.push_back(s);
  }

  // Cylinders
  for (auto &cylinder : cylinders)
  {
    geometric_primitive_msgs::msg::Cylinder c;
    c.id = cylinder.id;
    c.pose.position.x = cylinder.model.pose.position.x();
    c.pose.position.y = cylinder.model.pose.position.y();
    c.pose.position.z = cylinder.model.pose.position.z();
    c.pose.orientation.x = cylinder.model.pose.orientation.x();
    c.pose.orientation.y = cylinder.model.pose.orientation.y();
    c.pose.orientation.z = cylinder.model.pose.orientation.z();
    c.pose.orientation.w = cylinder.model.pose.orientation.w();
    c.radius = cylinder.model.radius;
    c.height = cylinder.model.height;
    msg_primitives.primitives.cylinders.push_back(c);
  }

  pub_primitives_->publish(msg_primitives);

  ///
  /// Visualisation
  ///

  // Visualise segment point clouds and the fitted geometric primitives, if desired
  if (parameters_.visualise)
  {
    // Clean up from previous callback
    viewer_->removeAllShapes();
    viewer_->removeAllPointClouds();

    // Create PointXYZL cloud of segments for visualisation
    pcl::PointCloud<pcl::PointXYZL>::Ptr cpc_labeled_cloud = supervoxel_clustering.getLabeledCloud();
    segmentation.relabelCloud(*cpc_labeled_cloud);

    // Add point cloud of the segments
    pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZL> point_cloud_color_handler(cpc_labeled_cloud, "label");
    viewer_->addPointCloud<pcl::PointXYZL>(cpc_labeled_cloud, point_cloud_color_handler, "segments");
    viewer_->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "segments");

    // Planes
    for (auto &plane : planes)
    {
      PointT first_point = segmented_point_clouds.at(plane.id)->points[0];
      pcl::ModelCoefficients coefficients;
      coefficients.values.push_back(plane._raw_coefficients[0]);
      coefficients.values.push_back(plane._raw_coefficients[1]);
      coefficients.values.push_back(plane._raw_coefficients[2]);
      coefficients.values.push_back(plane._raw_coefficients[3]);
      viewer_->addPlane(coefficients, first_point.x, first_point.y, first_point.z, "plane_" + std::to_string(plane.id));
    }

    // Spheres
    for (auto &sphere : spheres)
    {
      pcl::ModelCoefficients coefficients;
      coefficients.values.push_back(sphere._raw_coefficients[0]);
      coefficients.values.push_back(sphere._raw_coefficients[1]);
      coefficients.values.push_back(sphere._raw_coefficients[2]);
      coefficients.values.push_back(sphere._raw_coefficients[3]);
      viewer_->addSphere(coefficients, "sphere_" + std::to_string(sphere.id));
    }

    // Cylinders
    for (auto &cylinder : cylinders)
    {
      pcl::ModelCoefficients coefficients;
      coefficients.values.push_back(cylinder._raw_coefficients[0]);
      coefficients.values.push_back(cylinder._raw_coefficients[1]);
      coefficients.values.push_back(cylinder._raw_coefficients[2]);
      coefficients.values.push_back(cylinder._raw_coefficients[3]);
      coefficients.values.push_back(cylinder._raw_coefficients[4]);
      coefficients.values.push_back(cylinder._raw_coefficients[5]);
      coefficients.values.push_back(cylinder._raw_coefficients[6]);
      viewer_->addCylinder(coefficients, "cylinder_" + std::to_string(cylinder.id));
    }

    viewer_->spinOnce(50);
  }

  // Publish geometric primitives as a visualisation markers, if desired
  if (this->get_parameter("publish_markers").get_value<bool>())
  {
    visualization_msgs::msg::MarkerArray cleanup;
    visualization_msgs::msg::Marker cleanup_marker;
    cleanup_marker.header = msg_point_cloud->header;
    cleanup_marker.action = visualization_msgs::msg::Marker::DELETEALL;
    cleanup.markers.push_back(cleanup_marker);
    pub_markers_->publish(cleanup);

    visualization_msgs::msg::MarkerArray markers;
    visualization_msgs::msg::Marker default_marker;
    default_marker.header = msg_point_cloud->header;
    default_marker.action = visualization_msgs::msg::Marker::ADD;
    default_marker.ns = std::string(this->get_namespace()) + "geometric_primitives";
    default_marker.color.a = 0.75;

    // Planes
    {
      visualization_msgs::msg::Marker plane_marker = default_marker;
      plane_marker.ns += "/planes";
      plane_marker.type = visualization_msgs::msg::Marker::CUBE;
      for (auto &plane : planes)
      {
        plane_marker.id = plane.id;

        PointT first_point = segmented_point_clouds.at(plane.id)->points[0];
        plane_marker.pose.position.x = first_point.x;
        plane_marker.pose.position.y = first_point.y;
        plane_marker.pose.position.z = first_point.z;

        Eigen::Quaternion<double> quat;
        quat.setFromTwoVectors(Eigen::Vector3d::UnitZ(), plane.model.coeffs().head<3>().cast<double>());
        plane_marker.pose.orientation.x = quat.x();
        plane_marker.pose.orientation.y = quat.y();
        plane_marker.pose.orientation.z = quat.z();
        plane_marker.pose.orientation.w = quat.w();
        plane_marker.scale.x = 1.0;
        plane_marker.scale.y = 1.0;
        plane_marker.scale.z = 0.001;
        plane_marker.color.r = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        plane_marker.color.g = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        plane_marker.color.b = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        markers.markers.push_back(plane_marker);
      }
    }

    // Spheres
    {
      visualization_msgs::msg::Marker sphere_marker = default_marker;
      sphere_marker.ns += "/spheres";
      sphere_marker.type = visualization_msgs::msg::Marker::SPHERE;
      for (auto &sphere : spheres)
      {
        sphere_marker.id = sphere.id;

        sphere_marker.pose.position.x = sphere.model.centre[0];
        sphere_marker.pose.position.y = sphere.model.centre[1];
        sphere_marker.pose.position.z = sphere.model.centre[2];
        sphere_marker.scale.x =
            sphere_marker.scale.y =
                sphere_marker.scale.z = 2 * sphere.model.radius;
        sphere_marker.color.r = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        sphere_marker.color.g = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        sphere_marker.color.b = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        markers.markers.push_back(sphere_marker);
      }
    }

    // Cylinders
    {
      visualization_msgs::msg::Marker cylinder_marker = default_marker;
      cylinder_marker.ns += "/cylinders";
      cylinder_marker.type = visualization_msgs::msg::Marker::CYLINDER;
      for (auto &cylinder : cylinders)
      {
        cylinder_marker.id = cylinder.id;

        cylinder_marker.pose.position.x = cylinder.model.pose.position.x();
        cylinder_marker.pose.position.y = cylinder.model.pose.position.y();
        cylinder_marker.pose.position.z = cylinder.model.pose.position.z();

        cylinder_marker.pose.orientation.x = cylinder.model.pose.orientation.x();
        cylinder_marker.pose.orientation.y = cylinder.model.pose.orientation.y();
        cylinder_marker.pose.orientation.z = cylinder.model.pose.orientation.z();
        cylinder_marker.pose.orientation.w = cylinder.model.pose.orientation.w();
        cylinder_marker.scale.x =
            cylinder_marker.scale.y = 2 * cylinder.model.radius;
        cylinder_marker.scale.z = cylinder.model.height;
        cylinder_marker.color.r = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        cylinder_marker.color.g = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        cylinder_marker.color.b = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        markers.markers.push_back(cylinder_marker);
      }
    }
    pub_markers_->publish(markers);
  }
}

void GeometricPrimitiveFitting::declare_parameters()
{
  // Run-time immutable - not part of `update_parameters()`
  parameters_.publish_markers = this->declare_parameter<bool>("publish_markers", true);
  parameters_.visualise = this->declare_parameter<bool>("visualise", false);

  // Mutable
  parameters_.downsample_input = this->declare_parameter<bool>("downsampling.enable", true);
  parameters_.voxel_leaf_size_xy = this->declare_parameter<float>("downsampling.leaf_size.xy", 0.0025);
  parameters_.voxel_leaf_size_z = this->declare_parameter<float>("downsampling.leaf_size.z", 0.0025);

  parameters_.use_single_camera_transform = this->declare_parameter<bool>("supervoxel.use_single_camera_transform", true);
  parameters_.voxel_resolution = this->declare_parameter<float>("supervoxel.voxel_resolution", 0.01);
  parameters_.seed_resolution = this->declare_parameter<float>("supervoxel.seed_resolution", 0.05);
  parameters_.spatial_importance = this->declare_parameter<float>("supervoxel.importance.spatial", 1.0);
  parameters_.color_importance = this->declare_parameter<float>("supervoxel.importance.color", 0.25);
  parameters_.enable_normals = this->declare_parameter<bool>("supervoxel.normal.enable", true);
  parameters_.normal_search_radius = this->declare_parameter<float>("supervoxel.normal.search_radius", 0.025);
  parameters_.normal_importance = this->declare_parameter<float>("supervoxel.normal.importance", 4.0);
  parameters_.enable_supervoxel_refinement = this->declare_parameter<bool>("supervoxel.refinement.enable", true);
  parameters_.supervoxel_refinement_iterations = this->declare_parameter<uint8_t>("supervoxel.refinement.iterations", 5);

  parameters_.concavity_tolerance_threshold = this->declare_parameter<float>("segmentation.lccp.concavity_tolerance_threshold", 20.0);
  parameters_.smoothness_threshold = this->declare_parameter<float>("segmentation.lccp.smoothness_threshold", 0.25);
  parameters_.min_segment_size = this->declare_parameter<uint16_t>("segmentation.lccp.min_segment_size", 1);
  parameters_.k_factor = this->declare_parameter<uint8_t>("segmentation.lccp.k_factor", 1);
  parameters_.sanity_criterion = this->declare_parameter<bool>("segmentation.lccp.sanity_criterion", true);

  parameters_.max_cuts = this->declare_parameter<uint8_t>("segmentation.cpc.max_cuts", 10);
  parameters_.cutting_min_segments = this->declare_parameter<uint16_t>("segmentation.cpc.cutting_min_segments", 500);
  parameters_.cutting_min_score = this->declare_parameter<float>("segmentation.cpc.cutting_min_score", 0.15);
  parameters_.local_constrain = this->declare_parameter<bool>("segmentation.cpc.local_constrain", true);
  parameters_.directed_cutting = this->declare_parameter<bool>("segmentation.cpc.directed_cutting", true);
  parameters_.clean_cutting = this->declare_parameter<bool>("segmentation.cpc.clean_cutting", true);
  parameters_.ransac_iterations = this->declare_parameter<uint16_t>("segmentation.cpc.ransac_iterations", 1000);

  parameters_.sac_min_segment_size = this->declare_parameter<uint16_t>("sample_consensus.min_segment_size", 50);
  parameters_.plane.enable = this->declare_parameter<bool>("sample_consensus.plane.enable", true);
  parameters_.plane.distance_threshold = this->declare_parameter<float>("sample_consensus.plane.distance_threshold", 0.005);
  parameters_.plane.probability = this->declare_parameter<float>("sample_consensus.plane.probability", 0.99);
  parameters_.plane.max_iterations = this->declare_parameter<uint16_t>("sample_consensus.plane.max_iterations", 1000);
  parameters_.plane.min_inlier_proportion = this->declare_parameter<float>("sample_consensus.plane.min_inlier_proportion", 0.85);
  parameters_.plane.merge_precision = this->declare_parameter<float>("sample_consensus.plane.merge_precision", 0.25);
  parameters_.sphere.enable = this->declare_parameter<bool>("sample_consensus.sphere.enable", false);
  parameters_.sphere.min_radius = this->declare_parameter<float>("sample_consensus.sphere.min_radius", 0.01);
  parameters_.sphere.max_radius = this->declare_parameter<float>("sample_consensus.sphere.max_radius", 0.075);
  parameters_.sphere.distance_threshold = this->declare_parameter<float>("sample_consensus.sphere.distance_threshold", 0.005);
  parameters_.sphere.probability = this->declare_parameter<float>("sample_consensus.sphere.probability", 0.99);
  parameters_.sphere.max_iterations = this->declare_parameter<uint16_t>("sample_consensus.sphere.max_iterations", 1000);
  parameters_.sphere.min_inlier_proportion = this->declare_parameter<float>("sample_consensus.sphere.min_inlier_proportion", 0.75);
  parameters_.cylinder.enable = this->declare_parameter<bool>("sample_consensus.cylinder.enable", true);
  parameters_.cylinder.min_radius = this->declare_parameter<float>("sample_consensus.cylinder.min_radius", 0.007);
  parameters_.cylinder.max_radius = this->declare_parameter<float>("sample_consensus.cylinder.max_radius", 0.075);
  parameters_.cylinder.distance_threshold = this->declare_parameter<float>("sample_consensus.cylinder.distance_threshold", 0.005);
  parameters_.cylinder.probability = this->declare_parameter<float>("sample_consensus.cylinder.probability", 0.99);
  parameters_.cylinder.max_iterations = this->declare_parameter<uint16_t>("sample_consensus.cylinder.max_iterations", 1000);
  parameters_.cylinder.min_inlier_proportion = this->declare_parameter<float>("sample_consensus.cylinder.min_inlier_proportion", 0.7);
}

void GeometricPrimitiveFitting::update_parameters()
{
  parameters_.downsample_input = this->get_parameter("downsampling.enable").get_value<bool>();
  parameters_.voxel_leaf_size_xy = this->get_parameter("downsampling.leaf_size.xy").get_value<float>();
  parameters_.voxel_leaf_size_z = this->get_parameter("downsampling.leaf_size.z").get_value<float>();

  parameters_.use_single_camera_transform = this->get_parameter("supervoxel.use_single_camera_transform").get_value<bool>();
  parameters_.voxel_resolution = this->get_parameter("supervoxel.voxel_resolution").get_value<float>();
  parameters_.seed_resolution = this->get_parameter("supervoxel.seed_resolution").get_value<float>();
  parameters_.spatial_importance = this->get_parameter("supervoxel.importance.spatial").get_value<float>();
  parameters_.color_importance = this->get_parameter("supervoxel.importance.color").get_value<float>();
  parameters_.enable_normals = this->get_parameter("supervoxel.normal.enable").get_value<bool>();
  parameters_.normal_search_radius = this->get_parameter("supervoxel.normal.search_radius").get_value<float>();
  parameters_.normal_importance = this->get_parameter("supervoxel.normal.importance").get_value<float>();
  parameters_.enable_supervoxel_refinement = this->get_parameter("supervoxel.refinement.enable").get_value<bool>();
  parameters_.supervoxel_refinement_iterations = this->get_parameter("supervoxel.refinement.iterations").get_value<uint8_t>();

  parameters_.concavity_tolerance_threshold = this->get_parameter("segmentation.lccp.concavity_tolerance_threshold").get_value<float>();
  parameters_.smoothness_threshold = this->get_parameter("segmentation.lccp.smoothness_threshold").get_value<float>();
  parameters_.min_segment_size = this->get_parameter("segmentation.lccp.min_segment_size").get_value<uint16_t>();
  parameters_.k_factor = this->get_parameter("segmentation.lccp.k_factor").get_value<uint8_t>();
  parameters_.sanity_criterion = this->get_parameter("segmentation.lccp.sanity_criterion").get_value<bool>();

  parameters_.max_cuts = this->get_parameter("segmentation.cpc.max_cuts").get_value<uint8_t>();
  parameters_.cutting_min_segments = this->get_parameter("segmentation.cpc.cutting_min_segments").get_value<uint16_t>();
  parameters_.cutting_min_score = this->get_parameter("segmentation.cpc.cutting_min_score").get_value<float>();
  parameters_.local_constrain = this->get_parameter("segmentation.cpc.local_constrain").get_value<bool>();
  parameters_.directed_cutting = this->get_parameter("segmentation.cpc.directed_cutting").get_value<bool>();
  parameters_.clean_cutting = this->get_parameter("segmentation.cpc.clean_cutting").get_value<bool>();
  parameters_.ransac_iterations = this->get_parameter("segmentation.cpc.ransac_iterations").get_value<uint16_t>();

  parameters_.sac_min_segment_size = this->get_parameter("sample_consensus.min_segment_size").get_value<uint16_t>();
  parameters_.plane.enable = this->get_parameter("sample_consensus.plane.enable").get_value<bool>();
  parameters_.plane.distance_threshold = this->get_parameter("sample_consensus.plane.distance_threshold").get_value<float>();
  parameters_.plane.probability = this->get_parameter("sample_consensus.plane.probability").get_value<float>();
  parameters_.plane.max_iterations = this->get_parameter("sample_consensus.plane.max_iterations").get_value<uint16_t>();
  parameters_.plane.min_inlier_proportion = this->get_parameter("sample_consensus.plane.min_inlier_proportion").get_value<float>();
  parameters_.plane.merge_precision = this->get_parameter("sample_consensus.plane.merge_precision").get_value<float>();
  parameters_.sphere.enable = this->get_parameter("sample_consensus.sphere.enable").get_value<bool>();
  parameters_.sphere.min_radius = this->get_parameter("sample_consensus.sphere.min_radius").get_value<float>();
  parameters_.sphere.max_radius = this->get_parameter("sample_consensus.sphere.max_radius").get_value<float>();
  parameters_.sphere.distance_threshold = this->get_parameter("sample_consensus.sphere.distance_threshold").get_value<float>();
  parameters_.sphere.probability = this->get_parameter("sample_consensus.sphere.probability").get_value<float>();
  parameters_.sphere.max_iterations = this->get_parameter("sample_consensus.sphere.max_iterations").get_value<uint16_t>();
  parameters_.sphere.min_inlier_proportion = this->get_parameter("sample_consensus.sphere.min_inlier_proportion").get_value<float>();
  parameters_.cylinder.enable = this->get_parameter("sample_consensus.cylinder.enable").get_value<bool>();
  parameters_.cylinder.min_radius = this->get_parameter("sample_consensus.cylinder.min_radius").get_value<float>();
  parameters_.cylinder.max_radius = this->get_parameter("sample_consensus.cylinder.max_radius").get_value<float>();
  parameters_.cylinder.distance_threshold = this->get_parameter("sample_consensus.cylinder.distance_threshold").get_value<float>();
  parameters_.cylinder.probability = this->get_parameter("sample_consensus.cylinder.probability").get_value<float>();
  parameters_.cylinder.max_iterations = this->get_parameter("sample_consensus.cylinder.max_iterations").get_value<uint16_t>();
  parameters_.cylinder.min_inlier_proportion = this->get_parameter("sample_consensus.cylinder.min_inlier_proportion").get_value<float>();
}

////////////
/// MAIN ///
////////////

/// Main function that initiates an object of `GeometricPrimitiveFitting` class as the core of this node.
int main(int argc, char *argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<GeometricPrimitiveFitting>());
  rclcpp::shutdown();
  return EXIT_SUCCESS;
}
