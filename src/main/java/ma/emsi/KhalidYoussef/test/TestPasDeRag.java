package ma.emsi.KhalidYoussef.test;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.loader.FileSystemDocumentLoader;
import dev.langchain4j.data.document.parser.apache.tika.ApacheTikaDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.googleai.GoogleAiEmbeddingModel;
import dev.langchain4j.model.googleai.GoogleAiGeminiChatModel;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.rag.query.Query;
import dev.langchain4j.rag.query.router.QueryRouter;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import ma.emsi.KhalidYoussef.Assistant;

import java.nio.file.Path;
import java.util.*;
import java.util.logging.ConsoleHandler;
import java.util.logging.Level;
import java.util.logging.Logger;
public class TestPasDeRag {
    private static void configureLogger() {
        Logger packageLogger = Logger.getLogger("dev.langchain4j");
        packageLogger.setLevel(Level.FINE);

        ConsoleHandler handler = new ConsoleHandler();
        handler.setLevel(Level.FINE);

        packageLogger.setUseParentHandlers(false);
        packageLogger.addHandler(handler);
    }
    // ----> phase 1:
    // Méthode d'ingestion simple (charge + split + embeddings -> store)
    private static void ingest(
            Path path, EmbeddingStore<TextSegment> store,
            EmbeddingModel embModel) {
        Document document = FileSystemDocumentLoader.loadDocument(
                path, new ApacheTikaDocumentParser());
        List<TextSegment> segments = DocumentSplitters.recursive(500, 100).split(document);

        if (segments == null || segments.isEmpty()) {
            System.out.println("Aucun segment trouvé pour : " + path);
            return;
        }

        for (TextSegment segment : segments) {
            Embedding embedding = embModel.embed(segment.text()).content();
            store.add(embedding, segment);
        }

        System.out.println("Ingest terminé pour : " + path + " (segments = " + segments.size() + ")");
    }

    public static void main(String[] args) {
        configureLogger();

        String apiKey = System.getenv("GEMINI_KEY");
        if (apiKey == null || apiKey.isBlank()) {
            System.err.println(
                    "ERREUR: la variable d'environnement GEMINI_KEY n'est pas définie.");
            return;
        }

        // Chat model (Gemini) — utilisé POUR LE ROUTAGE ET L'ASSISTANT final
        ChatModel chatModel = GoogleAiGeminiChatModel.builder()
                .apiKey(apiKey)
                .modelName("gemini-2.5-flash")
                .temperature(0.5)
                .build();

        // Embedding model
        EmbeddingModel embeddingModel = GoogleAiEmbeddingModel.builder()
                .apiKey(apiKey)
                .modelName("text-embedding-004")
                .build();

        // PHASE 1 : INGESTION -> un seul store (support sur le RAG)
        EmbeddingStore<TextSegment> storeRag = new InMemoryEmbeddingStore<>();
        Path ragPdf = Path.of("src/main/resources/FinetuningEtRAG.pdf");
        ingest(ragPdf, storeRag, embeddingModel);

        // PHASE 2 : ContentRetriever
        ContentRetriever retrieverRag = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(storeRag)
                .embeddingModel(embeddingModel)
                .maxResults(3)
                .build();
        //QueryRouter personnalisé
        QueryRouter queryRouter = new QueryRouter() {
            @Override
            public Collection<ContentRetriever> route(Query query) {
                String prompt = "Est-ce que la requête \"" + query + "\" porte sur les concepts du RAG ? "
                        + "Réponds seulement par 'oui', 'non' ou 'peut-être'.";
                // Poser la question au LM directement
                ChatRequest chatRequest = ChatRequest.builder()
                        .messages(UserMessage.from(prompt))
                        .build();
                String lmReponse = chatModel.chat(chatRequest)
                        .aiMessage()
                        .text()
                        .toLowerCase();
                if (lmReponse.contains("oui") || lmReponse.contains("peut-être")) {
                    return List.of(retrieverRag);
                } else {
                    return Collections.emptyList();
                }
            }
        };
        // 4) RetrievalAugmentor
        RetrievalAugmentor augmentor = DefaultRetrievalAugmentor.builder()
                .queryRouter(queryRouter)
                .build();

        // 5) Assistant RAG
        Assistant assistant = AiServices.builder(Assistant.class)
                .chatModel(chatModel)
                .retrievalAugmentor(augmentor)
                .build();

        // Test interactif : taper "Bonjour" d'abord, puis poser une question sur le PDF
        try (Scanner scanner = new Scanner(System.in)) {
            System.out.println("Assistant prêt. Tape 'fin' pour quitter.");
            while (true) {
                System.out.print("[USER] > ");
                String question = scanner.nextLine();
                if (question == null) break;
                question = question.trim();
                if (question.isEmpty()) continue;
                if ("fin".equalsIgnoreCase(question) || "exit".equalsIgnoreCase(question)) break;

                // Appel à l'assistant. Selon ta version, assistant.chat(...) peut renvoyer un objet; j'ai supposé String.
                String reponse = assistant.chat(question);
                System.out.println("[ASSISTANT] : " + reponse);
                System.out.println("--------------------------------------------------");
            }
        }

        System.out.println("Fin TestNoRAG.");
    }
}
