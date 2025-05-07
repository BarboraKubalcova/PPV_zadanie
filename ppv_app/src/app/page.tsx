'use client';

import { useState, useCallback } from "react";
import Image from "next/image";
import styles from "./page.module.css";

export default function Home() {
  const [image, setImage] = useState<string | null>(null);
  const [message, setMessage] = useState("");
  const [isDragging, setIsDragging] = useState(false);

  const handleDragOver = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);

    const files = e.dataTransfer.files;
    if (files && files[0]) {
      const file = files[0];
      const fileReader = new FileReader();
      
      fileReader.onload = (event) => {
        if (event.target && typeof event.target.result === "string") {
          setImage(event.target.result);
        }
      };
      
      fileReader.readAsDataURL(file);
    }
  }, []);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      const fileReader = new FileReader();
      
      fileReader.onload = (event) => {
        if (event.target && typeof event.target.result === "string") {
          setImage(event.target.result);
        }
      };
      
      fileReader.readAsDataURL(file);
    }
  };

  const handleRemoveImage = () => {
    setImage(null);
  };

  const handleMessageChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setMessage(e.target.value);
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    // Process message here when implementing functionality
    console.log({ image, message });
  };

  return (
    <div className={styles.page}>
      <main className={styles.main}>
        <h1 className={styles.title}>PPV App</h1>

        <div 
          className={`${styles.dropzone} ${isDragging ? styles.dragging : ""}`}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
        >
          {image ? (
            <div className={styles.imagePreview}>
              <img src={image} alt="Uploaded preview" className={styles.uploadedImage} />
              <button 
                type="button" 
                onClick={handleRemoveImage} 
                className={styles.removeButton}
                aria-label="Remove image"
              >
                <span>×</span>
              </button>
            </div>
          ) : (
            <div className={styles.uploadPlaceholder}>
              <Image 
                src="/window.svg" 
                alt="Upload icon" 
                width={40} 
                height={40} 
                className={styles.uploadIcon}
              />
              <p>Drag & drop image here or</p>
              <label className={styles.fileInputLabel}>
                Browse Files
                <input 
                  type="file" 
                  accept="image/*" 
                  onChange={handleFileChange} 
                  className={styles.fileInput}
                />
              </label>
            </div>
          )}
        </div>

        <form onSubmit={handleSubmit} className={styles.messageForm}>
          <input
            type="text"
            value={message}
            onChange={handleMessageChange}
            placeholder="Enter your prompt"
            className={styles.messageInput}
          />
          <button type="submit" className={styles.submitButton}>
            Send
          </button>
        </form>
        
        <div className={styles.responseSection}>
          <h2 className={styles.responseTitle}>Response of the model:</h2>
          <div className={styles.responseContent}>
            default
          </div>
        </div>
      </main>
    </div>
  );
}
