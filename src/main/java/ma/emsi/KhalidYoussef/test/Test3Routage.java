package ma.emsi.KhalidYoussef.test;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.loader.FileSystemDocumentLoader;
import dev.langchain4j.data.document.parser.apache.tika.ApacheTikaDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.memory.ChatMemory;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.googleai.GoogleAiEmbeddingModel;
import dev.langchain4j.model.googleai.GoogleAiGeminiChatModel;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.rag.query.router.LanguageModelQueryRouter;
import dev.langchain4j.rag.query.router.QueryRouter;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import ma.emsi.KhalidYoussef.Assistant;

import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;
import java.util.Scanner;
import java.util.List;
import java.util.logging.ConsoleHandler;
import java.util.logging.Level;
import java.util.logging.Logger;

public class Test3Routage {

    // Active logger pour voir le routage (LLM queries)
    private static void configureLogger() {
        Logger packageLogger = Logger.getLogger("dev.langchain4j");
        packageLogger.setLevel(Level.FINE);

        ConsoleHandler handler = new ConsoleHandler();
        handler.setLevel(Level.FINE);

        packageLogger.setUseParentHandlers(false);
        packageLogger.addHandler(handler);

        System.out.println("--- Logging LangChain4j activé (niveau FINE) ---");
    }

    private static void ingest(Path path, EmbeddingStore<TextSegment> store, EmbeddingModel embModel) {
        Document document = FileSystemDocumentLoader.loadDocument(path, new ApacheTikaDocumentParser());
        List<TextSegment> segments = DocumentSplitters.recursive(300, 100).split(document);

        if (segments == null || segments.isEmpty()) {
            System.out.println("Aucun segment trouvé pour : " + path);
            return;
        }

        for (TextSegment segment : segments) {
            // Note : selon ta version de langchain4j, embModel.embed(...) peut renvoyer un objet différent.
            // Ici on suppose embModel.embed(segment.text()).content() renvoie un Embedding.
            Embedding embedding = embModel.embed(segment.text()).content();

            // selon la signature du store, utilise store.add(embedding, segment) ou store.addAll(...)
            store.add(embedding, segment);
        }

        System.out.println("Ingest terminé pour : " + path + " (segments = " + segments.size() + ")");
    }

    public static void main(String[] args) {
        configureLogger();

        String apiKey = System.getenv("GEMINI_KEY");
        if (apiKey == null || apiKey.isBlank()) {
            System.err.println("ERREUR: la variable d'environnement GEMINI_KEY n'est pas définie.");
            return;
        }

        // Chat model (Gemini) utilisé pour le routage et pour l'assistant
        ChatModel chatModel = GoogleAiGeminiChatModel.builder()
                .apiKey(apiKey)
                .modelName("gemini-2.5-flash")
                .temperature(0.3)
                .build();

        // Embedding model (Gemini embeddings / Google embeddings)
        EmbeddingModel embeddingModel = GoogleAiEmbeddingModel.builder()
                .apiKey(apiKey)
                .modelName("text-embedding-004")
                .build();

        // Mémoire du chat (facultative mais demandée par l'énoncé)
        ChatMemory chatMemory = MessageWindowChatMemory.withMaxMessages(10);

        // PHASE 1 : ingestion — 2 EmbeddingStores séparés
        EmbeddingStore<TextSegment> storeCours = new InMemoryEmbeddingStore<>();
        EmbeddingStore<TextSegment> storeAutre = new InMemoryEmbeddingStore<>();

        // Adapte les chemins si nécessaire (ou utilise getResource si tu préfères ressources du classpath)
        ingest(Path.of("src/main/resources/FinetuningEtRAG.pdf"), storeCours, embeddingModel);
        ingest(Path.of("src/main/resources/dynamiqueDesGroupes.pdf"), storeAutre, embeddingModel);

        // PHASE 2 : création des ContentRetrievers (un par store)
        ContentRetriever retrieverCours = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(storeCours)
                .embeddingModel(embeddingModel)
                .maxResults(3)
                .minScore(0.0)
                .build();

        ContentRetriever retrieverAutre = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(storeAutre)
                .embeddingModel(embeddingModel)
                .maxResults(3)
                .minScore(0.0)
                .build();

        // Descriptions (Map<ContentRetriever, String>) — l'input lu par le LLM pour décider
        Map<ContentRetriever, String> descriptions = new HashMap<>();
        descriptions.put(retrieverCours, "Documents sur le RAG, fine-tuning et architectures d'IA.");
        descriptions.put(retrieverAutre, "Documents sur la dynamique des groupe et les 3 pères fondateurs");

        // QueryRouter basé sur le LLM (Gemini)
        QueryRouter router = new LanguageModelQueryRouter(chatModel, descriptions);

        // RetrievalAugmentor qui utilisera le QueryRouter pour décider quelles sources interroger
        RetrievalAugmentor augmentor = DefaultRetrievalAugmentor.builder()
                .queryRouter(router)
                .build();

        // Création de l'assistant RAG
        Assistant assistant = AiServices.builder(Assistant.class)
                .chatModel(chatModel)
                .retrievalAugmentor(augmentor)
                .chatMemory(chatMemory)
                .build();

        // Boucle interactive
        try (Scanner scanner = new Scanner(System.in)) {
            System.out.println("Assistant prêt. Pose des questions: ");
            while (true) {
                System.out.print("[USER] > ");
                String question = scanner.nextLine();
                if (question == null) break;
                question = question.trim();
                if (question.isEmpty()) continue;
                if ("fin".equalsIgnoreCase(question) ||
                        "exit".equalsIgnoreCase(question) ||
                        "quitter".equalsIgnoreCase(question) ||
                        "bye".equalsIgnoreCase(question)) {
                    break;
                }

                // Appel à l'assistant.
                String reponse = assistant.chat(question);

                System.out.println("[ASSISTANT] : " + reponse);
            }
        }

        System.out.println("Fin du TestRoutage.");
    }
}
