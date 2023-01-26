#include <fstream>
#include <algorithm>

#include <boost/filesystem.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <loguru/loguru.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "nerf-lab/data/camera_model.h"
#include "nerf-lab/data/colmap_dataset.h"

namespace nerf {

namespace fs = boost::filesystem;
using namespace Eigen;

ColmapDataset::ColmapDataset(fs::path root) {
    root_ = root;
    images_dir_ = root / "images";
    database_path_ = root_ / "database.db";

    cameras_file_ = root_ / "sparse" / "0" / "cameras.bin";
    images_file_ = root_ / "sparse" / "0" / "images.bin";
    points_txt_ = root_ / "sparse" / "0" / "points3D.bin";

    CHECK_F(fs::exists(images_dir_) && fs::is_directory(images_dir_),
            "%s does not exists or is not a directory", images_dir_.c_str());
    
    for (fs::path image_file : fs::directory_iterator(images_dir_)) {
        images_paths_.push_back(image_file);
    }

    std::sort(images_paths_.begin(), images_paths_.end());

    LOG_F(INFO, "Found %ld images in %s", images_paths_.size(), images_dir_.c_str());
    
    database_found_ = false;

    try_load_images_();
    try_load_cameras_();
}

cv::Mat ColmapDataset::load_image(int index) const {
    fs::path image_path = images_paths_[index];
    cv::Mat image = cv::imread(image_path.c_str());
    cv::cvtColor(image, image, cv::ColorConversionCodes::COLOR_BGR2RGB);

    LOG_F(INFO, "Loaded image %s", image_path.c_str());
    
    return image;
}

// void ColmapDataset::try_load_db_() {
//     if (!fs::exists(database_path_)) {
//         LOG_F(INFO, "No database file found: %s", database_path_.c_str());
//         return;
//     }

//     sqlite3* db;
//     sqlite3_open(database_path_.c_str(), &db);

//     CHECK_F(db != nullptr, "Cannot connect to colmap's database %s", database_path_.c_str());
//     LOG_F(INFO, "Loaded colmap's database by path: %s", database_path_.c_str());

//     sqlite3_stmt* stmt;

//     sqlite3_prepare_v2(db, "SELECT image_id, camera_id, name FROM images;", -1, &stmt, nullptr);
//     sqlite3_step(stmt);
//     int image_id = sqlite3_column_int(stmt, 0);
//     int camera_id = sqlite3_column_int(stmt, 1);

//     const unsigned char* image_name_c = sqlite3_column_text(stmt, 2);
//     std::string image_name((const char*)image_name_c);

//     image_name_to_id_[image_name] = image_id;
//     image_id_to_name_[image_id] = image_name;
//     image_to_camera_id_[image_id] = camera_id;

//     database_found_ = true;
// }

void ColmapDataset::try_load_images_() {
    if (!fs::exists(images_file_)) {
        LOG_F(INFO, "%s is not found", images_file_.c_str());
        return;
    }

    std::ifstream file(images_file_, std::ios::binary);
    
    int64_t num_images;
    file.read((char*)&num_images, sizeof(int64_t));

    for (int i = 0; i < num_images; ++i) {
        int32_t image_id;
        int32_t camera_id;
        char image_name[256];

        Quaterniond rotation;
        Vector3d translation;

        file.read((char*)&image_id, sizeof(int32_t));
        file.read((char*)&rotation.w(), sizeof(double));
        file.read((char*)&rotation.x(), sizeof(double));
        file.read((char*)&rotation.y(), sizeof(double));
        file.read((char*)&rotation.z(), sizeof(double));
        file.read((char*)&translation.x(), sizeof(double));
        file.read((char*)&translation.y(), sizeof(double));
        file.read((char*)&translation.z(), sizeof(double));
        file.read((char*)&camera_id, sizeof(int32_t));
        file.getline(image_name, 256, 0);

        int64 num_points;
        file.read((char*)&num_points, sizeof(int64_t));
        file.ignore(num_points * (2 * sizeof(double) + sizeof(int64_t)));

        image_name_to_id_[image_name] = image_id;
        image_id_to_name_[image_id] = image_name;
        image_to_camera_id_[image_id] = camera_id;

        std::shared_ptr<ImageHandle> image = std::make_shared<ImageHandle>(image_id, image_name);
        
        rotation = rotation.inverse();
        translation = -(rotation * translation);

        image->set_transform(translation.cast<float>(), rotation.cast<float>());

        if (has_cameras()) {
            image->set_camera(cameras_[camera_id]);
        }

        images_[image_id] = image;
    }
}

void ColmapDataset::try_load_cameras_() {
    using ModelType = Camera::ModelType;

    if (!fs::exists(cameras_file_)) {
        LOG_F(INFO, "%s is not found", images_file_.c_str());
        return;
    }

    std::ifstream file(cameras_file_, std::ios::binary);

    int64_t num_cameras;
    file.read((char*)&num_cameras, sizeof(int64_t));

    for (int i = 0; i < num_cameras; ++i) {
        int32_t camera_id;
        int32_t camera_model;
        int64_t width, height;
        double fx, fy;
        double cx, cy;
        double k1 = 0, k2 = 0, p1 = 0, p2 = 0, k3 = 0, k4 = 0, k5 = 0, k6 = 0;

        file.read((char*)&camera_id, sizeof(int32_t));
        file.read((char*)&camera_model, sizeof(int32_t));
        file.read((char*)&width, sizeof(int64_t));
        file.read((char*)&height, sizeof(int64_t));

        file.read((char*)&fx, sizeof(double));

        CHECK_F((ModelType)camera_model != ModelType::kFishEye &&
                (ModelType)camera_model <= ModelType::kLast, "Not supported camera model (enum: %d) yet", camera_model);

        if ((ModelType)camera_model == ModelType::kSimplePinhole ||
            (ModelType)camera_model == ModelType::kSimpleRadial ||
            (ModelType)camera_model == ModelType::kRadial) {
            fy = fx;
        } else {
            file.read((char*)&fy, sizeof(double));
        }

        file.read((char*)&cx, sizeof(double));
        file.read((char*)&cy, sizeof(double));

        // dist coefs
        if ((ModelType)camera_model != ModelType::kSimplePinhole &&
            (ModelType)camera_model != ModelType::kPinhole) {

            file.read((char*)&k1, sizeof(double));

            if ((ModelType)camera_model != ModelType::kSimpleRadial) {
                file.read((char*)&k2, sizeof(double));
            }

            if ((ModelType)camera_model == ModelType::kOpenCV ||
                (ModelType)camera_model == ModelType::kFullOpenCV) {
                
                file.read((char*)&p1, sizeof(double));
                file.read((char*)&p2, sizeof(double));
            }

            if ((ModelType)camera_model == ModelType::kFullOpenCV) {
                file.read((char*)&k3, sizeof(double));
                file.read((char*)&k4, sizeof(double));
                file.read((char*)&k5, sizeof(double));
                file.read((char*)&k6, sizeof(double));
            }
        }

        std::vector<float> dist_coefs{
            (float)k1, (float)k2, (float)p1, (float)p2, 
            (float)k3, (float)k4, (float)k5, (float)k6
        };

        Camera camera(width, height, fx, fy, cx, cy, dist_coefs, (ModelType)camera_model);
    }

    if (has_images()) {
        for (auto& [image_id, camera_id] : image_to_camera_id_) {
            images_[image_id]->set_camera(cameras_[camera_id]);
        }
    }
}

}