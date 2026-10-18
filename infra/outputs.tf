output "mlflow_bucket_name" {
  description = "Bucket name configured for MLflow artifact store"
  value       = aws_s3_bucket.mlflow_artifacts.id
}

output "mlflow_bucket_arn" {
  description = "ARN of the MLflow artifact store bucket"
  value       = aws_s3_bucket.mlflow_artifacts.arn
}

output "dvc_bucket_name" {
  description = "Bucket name configured for DVC remote storage"
  value       = aws_s3_bucket.dvc_storage.id
}

output "dvc_bucket_arn" {
  description = "ARN of the DVC remote storage bucket"
  value       = aws_s3_bucket.dvc_storage.arn
}

output "storage_endpoint" {
  description = "S3 Endpoint URL"
  value       = var.s3_endpoint
}
