from qdrant_client import QdrantClient

client = QdrantClient(host="localhost", port=6333)

client.delete_collection("nexus_documents")
client.delete_collection("nexus_chat_memory")

print("Collections deleted. They will be recreated on next app startup.")