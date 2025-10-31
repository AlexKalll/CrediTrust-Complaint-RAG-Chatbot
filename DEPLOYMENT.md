# 🚀 Deployment Guide for Streamlit Cloud

## ✅ What to Push to GitHub

### **Files to Include:**
- ✅ All source code (`src/`, `dashboards/`)
- ✅ `requirements.txt` - Python dependencies
- ✅ `data/sample_complaints.csv` - Sample data for demo
- ✅ `data/README.md` - Data instructions
- ✅ `.streamlit/config.toml` - Streamlit configuration
- ✅ `README.md` - Project documentation

### **Files NOT to Push:**
- ❌ `vectorstore/` - **DO NOT PUSH** (generated locally, very large)
- ❌ `data/filtered_complaints.csv` - Your actual data (privacy)
- ❌ `.env` - API keys and secrets
- ❌ `models/` - Large model files (if any)
- ❌ `__pycache__/` - Python cache

## 🔧 Required Dependencies

All dependencies are in `requirements.txt`:
- `faiss-cpu>=1.7.0` - Vector database (may take time to build on Streamlit Cloud)
- `plotly>=5.15.0` - Interactive charts
- `sentence-transformers>=2.2.0` - Embeddings
- `langchain>=0.0.300` - RAG framework
- All other dependencies listed in requirements.txt

## 📊 Vector Store Generation

### **For Local Development:**
1. Add your `filtered_complaints.csv` to `data/` folder
2. Run the embedding pipeline to generate `vectorstore/`
3. The vectorstore will be created locally in `vectorstore/` folder

### **For Streamlit Cloud:**
The app will run in **demo mode** without vectorstore. This is expected because:
- Vectorstore is large (100MB+) and shouldn't be in git
- It's generated from your specific data
- Streamlit Cloud has size limits for repositories

### **To Enable Full RAG Functionality on Streamlit Cloud:**
**Option 1: Generate Vectorstore on Streamlit Cloud** (Not Recommended)
- Requires running embedding pipeline on Streamlit Cloud
- Large files can cause deployment issues

**Option 2: Use External Vector Store** (Future Enhancement)
- Use a cloud vector database (Pinecone, Weaviate, etc.)
- Store embeddings externally
- Access via API

**Option 3: Accept Demo Mode** (Recommended for Now)
- App works with sample data
- Analytics dashboard functions fully
- Chat uses intelligent fallback responses
- Professional appearance maintained

## 🐛 Common Deployment Issues

### **Issue: "No module named 'faiss'"**
**Solution:** 
- ✅ `faiss-cpu` is already in requirements.txt
- ✅ First deployment may take 10-15 minutes (building faiss)
- ✅ If it fails, app runs in demo mode automatically

### **Issue: "Plotly not available"**
**Solution:**
- ✅ `plotly>=5.15.0` is in requirements.txt
- ✅ App falls back to text charts if plotly fails
- ✅ Should install automatically on redeploy

### **Issue: "DummyRAGPipeline() takes no arguments"**
**Solution:**
- ✅ Fixed! Dummy class now accepts initialization arguments
- ✅ Push latest code to resolve

## 📝 Deployment Checklist

Before pushing to GitHub:
- [ ] Verify `vectorstore/` is in `.gitignore`
- [ ] Verify `data/filtered_complaints.csv` is NOT in git (use sample data instead)
- [ ] Verify `.env` is in `.gitignore`
- [ ] Check `requirements.txt` includes all dependencies
- [ ] Test locally that app runs with sample data
- [ ] Commit and push changes

## 🎯 Expected Behavior After Deployment

### **What Will Work:**
✅ Analytics Dashboard (with sample data)
✅ Interactive Charts (if plotly installs)
✅ Text-based Charts (if plotly fails)
✅ Sample Data Display
✅ Chat Interface (demo mode with intelligent responses)
✅ All UI/UX features

### **What Requires Setup:**
⚠️ Full RAG Pipeline (needs vectorstore generation)
⚠️ Real Data Analysis (needs filtered_complaints.csv)
⚠️ Custom Model Files (if using local models)

## 💡 Recommendations

1. **For Demo/Portfolio:** Keep current setup - works perfectly
2. **For Production:** Consider cloud vector database solution
3. **For Large Data:** Use external data storage (S3, GCS, etc.)
4. **For API Keys:** Use Streamlit Cloud's secrets management

## 🚀 Quick Deploy Commands

```bash
# 1. Verify what will be pushed
git status

# 2. Add all changes
git add .

# 3. Commit with descriptive message
git commit -m "fix: resolve deployment issues and improve error handling"

# 4. Push to GitHub
git push origin main

# 5. Streamlit Cloud will auto-deploy
# Monitor deployment in Streamlit Cloud dashboard
```

## 📞 Need Help?

If deployment fails:
1. Check Streamlit Cloud logs (Manage app → Logs)
2. Verify all dependencies in requirements.txt
3. Check .gitignore excludes large files
4. Try restarting the app in Streamlit Cloud dashboard

