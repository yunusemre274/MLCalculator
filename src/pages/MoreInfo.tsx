import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { useNavigate } from "react-router-dom";
import { 
  Brain, 
  TrendingUp, 
  Network, 
  MessageSquare, 
  Eye, 
  Star,
  ArrowLeft,
  Code2,
  BookOpen
} from "lucide-react";

const MoreInfo = () => {
  const navigate = useNavigate();

  const mlSections = [
    {
      id: "classification",
      icon: Brain,
      title: "Classification Pro",
      subtitle: "Deep Learning & Pattern Recognition",
      description: "Classification is a supervised learning technique where the model learns to categorize data into predefined classes. It's widely used in image recognition, spam detection, and medical diagnosis.",
      technologies: ["CNN", "Transfer Learning", "Deep Learning", "Image Classification", "Text Classification"],
      keyPoints: [
        "Binary Classification: Two classes (e.g., spam/not spam)",
        "Multi-class Classification: Multiple classes (e.g., digit recognition 0-9)",
        "Convolutional Neural Networks (CNN): Specialized for image data",
        "Transfer Learning: Using pre-trained models like ResNet, VGG, or Inception"
      ],
      example: `# Simple Image Classification Example
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# Load data
digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(
    digits.data, digits.target, test_size=0.2
)

# Train model
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)

# Predict
accuracy = clf.score(X_test, y_test)
print(f"Accuracy: {accuracy:.2%}")`,
      useCases: [
        "Email Spam Detection",
        "Medical Image Diagnosis",
        "Sentiment Analysis",
        "Handwriting Recognition",
        "Fraud Detection"
      ]
    },
    {
      id: "prediction",
      icon: TrendingUp,
      title: "Prediction Engine",
      subtitle: "Time Series Forecasting & Regression",
      description: "Prediction engines forecast future values based on historical data patterns. They're essential for stock market analysis, sales forecasting, and weather prediction.",
      technologies: ["LSTM", "Time Series", "Regression", "ARIMA", "Prophet"],
      keyPoints: [
        "Time Series Analysis: Analyzing data points over time",
        "LSTM Networks: Long Short-Term Memory for sequential data",
        "Seasonality Detection: Identifying recurring patterns",
        "Trend Analysis: Understanding long-term directions"
      ],
      example: `# Time Series Prediction Example
from sklearn.linear_model import LinearRegression
import numpy as np

# Sample time series data
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 5, 4, 5])

# Train model
model = LinearRegression()
model.fit(X, y)

# Predict future values
future = np.array([[6], [7], [8]])
predictions = model.predict(future)
print(f"Predictions: {predictions}")`,
      useCases: [
        "Stock Price Forecasting",
        "Sales Prediction",
        "Weather Forecasting",
        "Energy Demand Prediction",
        "Traffic Flow Estimation"
      ]
    },
    {
      id: "clustering",
      icon: Network,
      title: "Cluster Analytics",
      subtitle: "Unsupervised Learning & Segmentation",
      description: "Clustering algorithms automatically group similar data points together without predefined labels. Perfect for customer segmentation, anomaly detection, and data exploration.",
      technologies: ["K-Means", "DBSCAN", "Hierarchical", "Gaussian Mixture", "Unsupervised Learning"],
      keyPoints: [
        "K-Means: Partition data into K distinct clusters",
        "DBSCAN: Density-based clustering for arbitrary shapes",
        "Hierarchical Clustering: Building tree-like cluster structures",
        "Silhouette Score: Measuring cluster quality"
      ],
      example: `# K-Means Clustering Example
from sklearn.cluster import KMeans
import numpy as np

# Sample data
X = np.array([[1, 2], [1.5, 1.8], [5, 8], 
              [8, 8], [1, 0.6], [9, 11]])

# Create and fit model
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X)

# Get cluster labels
labels = kmeans.labels_
print(f"Cluster labels: {labels}")`,
      useCases: [
        "Customer Segmentation",
        "Market Basket Analysis",
        "Document Organization",
        "Anomaly Detection",
        "Image Compression"
      ]
    },
    {
      id: "nlp",
      icon: MessageSquare,
      title: "NLP Analyzer",
      subtitle: "Natural Language Processing",
      description: "NLP enables machines to understand, interpret, and generate human language. It powers chatbots, translation services, and sentiment analysis systems.",
      technologies: ["BERT", "Transformers", "Word2Vec", "Sentiment Analysis", "Named Entity Recognition"],
      keyPoints: [
        "Tokenization: Breaking text into words or subwords",
        "Transformers: Attention-based neural networks (BERT, GPT)",
        "Sentiment Analysis: Detecting emotions in text",
        "Named Entity Recognition: Identifying people, places, organizations"
      ],
      example: `# Sentiment Analysis Example
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Sample data
texts = ["I love this product!", "Terrible experience", 
         "Amazing quality", "Worst purchase ever"]
labels = [1, 0, 1, 0]  # 1=positive, 0=negative

# Vectorize text
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

# Train model
model = MultinomialNB()
model.fit(X, labels)

# Predict sentiment
new_text = vectorizer.transform(["This is great!"])
sentiment = model.predict(new_text)
print(f"Sentiment: {'Positive' if sentiment[0] else 'Negative'}")`,
      useCases: [
        "Chatbot Development",
        "Sentiment Analysis",
        "Machine Translation",
        "Text Summarization",
        "Question Answering Systems"
      ]
    },
    {
      id: "vision",
      icon: Eye,
      title: "Vision Recognition",
      subtitle: "Computer Vision & Object Detection",
      description: "Computer vision enables machines to interpret and understand visual information from the world. It's used in autonomous vehicles, facial recognition, and medical imaging.",
      technologies: ["ResNet", "YOLO", "CNN", "Object Detection", "Image Segmentation"],
      keyPoints: [
        "Object Detection: Identifying and locating objects in images",
        "Image Segmentation: Pixel-level classification",
        "Facial Recognition: Identifying individuals from images",
        "ResNet/VGG: Deep convolutional architectures"
      ],
      example: `# Simple Image Feature Detection
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Simulated image features (height, width, intensity)
X = np.array([[100, 50, 200], [120, 60, 210], 
              [30, 80, 50], [25, 85, 45]])
y = np.array([1, 1, 0, 0])  # 1=cat, 0=dog

# Train classifier
clf = RandomForestClassifier()
clf.fit(X, y)

# Predict
new_image = np.array([[110, 55, 205]])
prediction = clf.predict(new_image)
print(f"Detected: {'Cat' if prediction[0] else 'Dog'}")`,
      useCases: [
        "Autonomous Vehicles",
        "Facial Recognition",
        "Medical Image Analysis",
        "Quality Control in Manufacturing",
        "Augmented Reality"
      ]
    },
    {
      id: "recommendation",
      icon: Star,
      title: "Recommendation System",
      subtitle: "Collaborative Filtering & Personalization",
      description: "Recommendation systems predict user preferences and suggest relevant items. They're the backbone of e-commerce, streaming services, and content platforms.",
      technologies: ["Collaborative Filtering", "Matrix Factorization", "Content-Based", "Hybrid Systems", "Neural Networks"],
      keyPoints: [
        "Collaborative Filtering: User-item interaction patterns",
        "Content-Based Filtering: Item feature similarity",
        "Matrix Factorization: SVD and ALS techniques",
        "Hybrid Approaches: Combining multiple methods"
      ],
      example: `# Simple Recommendation System
from sklearn.neighbors import NearestNeighbors
import numpy as np

# User-item rating matrix (users x items)
ratings = np.array([
    [5, 3, 0, 1],
    [4, 0, 0, 1],
    [1, 1, 0, 5],
    [1, 0, 0, 4],
])

# Find similar users
model = NearestNeighbors(metric='cosine')
model.fit(ratings)

# Get recommendations for user 0
distances, indices = model.kneighbors([ratings[0]], n_neighbors=3)
print(f"Similar users: {indices[0]}")`,
      useCases: [
        "E-commerce Product Recommendations",
        "Movie/Music Streaming Suggestions",
        "Content Discovery Platforms",
        "Personalized News Feeds",
        "Dating App Matches"
      ]
    }
  ];

  return (
    <div className="min-h-screen bg-background">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="mb-8">
          <Button 
            variant="ghost" 
            onClick={() => navigate("/")}
            className="mb-4"
          >
            <ArrowLeft className="mr-2 h-4 w-4" />
            Back to Home
          </Button>
          
          <div className="text-center space-y-2">
            <h1 className="text-4xl font-bold bg-gradient-to-r from-primary to-primary/60 bg-clip-text text-transparent">
              Machine Learning Guide
            </h1>
            <p className="text-xl text-muted-foreground max-w-3xl mx-auto">
              Comprehensive guide to understanding machine learning algorithms, techniques, and their real-world applications
            </p>
          </div>
        </div>

        {/* Main Content */}
        <Tabs defaultValue="classification" className="w-full">
          <TabsList className="grid grid-cols-3 lg:grid-cols-6 gap-2 h-auto p-2 mb-8">
            {mlSections.map((section) => (
              <TabsTrigger 
                key={section.id} 
                value={section.id}
                className="flex flex-col items-center gap-1 p-3"
              >
                <section.icon className="h-5 w-5" />
                <span className="text-xs">{section.title.split(" ")[0]}</span>
              </TabsTrigger>
            ))}
          </TabsList>

          {mlSections.map((section) => (
            <TabsContent key={section.id} value={section.id} className="space-y-6">
              {/* Overview Card */}
              <Card>
                <CardHeader>
                  <div className="flex items-start gap-4">
                    <div className="p-3 rounded-lg bg-primary/10">
                      <section.icon className="h-8 w-8 text-primary" />
                    </div>
                    <div className="flex-1">
                      <CardTitle className="text-3xl mb-2">{section.title}</CardTitle>
                      <CardDescription className="text-base">{section.subtitle}</CardDescription>
                    </div>
                  </div>
                </CardHeader>
                <CardContent className="space-y-6">
                  <div>
                    <h3 className="text-xl font-semibold mb-3 flex items-center gap-2">
                      <BookOpen className="h-5 w-5" />
                      Overview
                    </h3>
                    <p className="text-muted-foreground leading-relaxed">
                      {section.description}
                    </p>
                  </div>

                  {/* Technologies */}
                  <div>
                    <h3 className="text-lg font-semibold mb-3">Key Technologies</h3>
                    <div className="flex flex-wrap gap-2">
                      {section.technologies.map((tech, idx) => (
                        <Badge key={idx} variant="secondary" className="text-sm px-3 py-1">
                          {tech}
                        </Badge>
                      ))}
                    </div>
                  </div>

                  {/* Key Points */}
                  <div>
                    <h3 className="text-lg font-semibold mb-3">Key Concepts</h3>
                    <ul className="space-y-2">
                      {section.keyPoints.map((point, idx) => (
                        <li key={idx} className="flex items-start gap-2">
                          <span className="text-primary mt-1">•</span>
                          <span className="text-muted-foreground">{point}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                </CardContent>
              </Card>

              {/* Code Example Card */}
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Code2 className="h-5 w-5" />
                    Code Example
                  </CardTitle>
                  <CardDescription>
                    Practical implementation demonstrating the core concepts
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <pre className="bg-muted p-4 rounded-lg overflow-x-auto">
                    <code className="text-sm">{section.example}</code>
                  </pre>
                </CardContent>
              </Card>

              {/* Use Cases Card */}
              <Card>
                <CardHeader>
                  <CardTitle>Real-World Applications</CardTitle>
                  <CardDescription>
                    Common use cases and industry applications
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                    {section.useCases.map((useCase, idx) => (
                      <div 
                        key={idx}
                        className="flex items-center gap-2 p-3 rounded-lg bg-muted/50 hover:bg-muted transition-colors"
                      >
                        <div className="w-2 h-2 rounded-full bg-primary" />
                        <span className="text-sm">{useCase}</span>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>

              {/* CTA */}
              <Card className="bg-gradient-to-r from-primary/10 to-primary/5 border-primary/20">
                <CardContent className="pt-6">
                  <div className="flex flex-col md:flex-row items-center justify-between gap-4">
                    <div>
                      <h3 className="text-xl font-semibold mb-2">Ready to try it yourself?</h3>
                      <p className="text-muted-foreground">
                        Upload your dataset and train models with our interactive dashboard
                      </p>
                    </div>
                    <Button 
                      size="lg"
                      onClick={() => navigate("/dashboard")}
                      className="whitespace-nowrap"
                    >
                      Go to Dashboard
                    </Button>
                  </div>
                </CardContent>
              </Card>
            </TabsContent>
          ))}
        </Tabs>
      </div>
    </div>
  );
};

export default MoreInfo;
