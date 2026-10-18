# Storage Infrastructure for MLflow Artifacts and DVC Data Versioning

resource "aws_s3_bucket" "mlflow_artifacts" {
  bucket = var.mlflow_bucket_name

  tags = {
    Name        = "MLflow Artifact Store"
    Environment = var.environment
    Project     = "Arabic-Legal-RAG"
    ManagedBy   = "Terraform"
  }
}

resource "aws_s3_bucket_versioning" "mlflow_versioning" {
  bucket = aws_s3_bucket.mlflow_artifacts.id

  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_public_access_block" "mlflow_access" {
  bucket = aws_s3_bucket.mlflow_artifacts.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket" "dvc_storage" {
  bucket = var.dvc_bucket_name

  tags = {
    Name        = "DVC Remote Storage"
    Environment = var.environment
    Project     = "Arabic-Legal-RAG"
    ManagedBy   = "Terraform"
  }
}

resource "aws_s3_bucket_versioning" "dvc_versioning" {
  bucket = aws_s3_bucket.dvc_storage.id

  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_public_access_block" "dvc_access" {
  bucket = aws_s3_bucket.dvc_storage.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}
