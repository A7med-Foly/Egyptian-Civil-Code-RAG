variable "aws_region" {
  description = "AWS region or default region for MinIO"
  type        = string
  default     = "us-east-1"
}

variable "s3_endpoint" {
  description = "S3 Endpoint URL (e.g. MinIO http://localhost:9000 or AWS S3)"
  type        = string
  default     = "http://localhost:9000"
}

variable "aws_access_key" {
  description = "Access key for S3 / MinIO storage"
  type        = string
  default     = "minioadmin"
  sensitive   = true
}

variable "aws_secret_key" {
  description = "Secret key for S3 / MinIO storage"
  type        = string
  default     = "minioadmin"
  sensitive   = true
}

variable "mlflow_bucket_name" {
  description = "Bucket name for MLflow run artifacts and serialized models"
  type        = string
  default     = "legal-rag-mlflow-artifacts"
}

variable "dvc_bucket_name" {
  description = "Bucket name for DVC remote versioned datasets"
  type        = string
  default     = "legal-rag-dvc-storage"
}

variable "environment" {
  description = "Deployment environment name"
  type        = string
  default     = "production"
}
