package ma.emsi.KhalidYoussef.test;

import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.googleai.GoogleAiGeminiChatModel;

import java.net.URISyntaxException;
import java.net.URL;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.logging.ConsoleHandler;
import java.util.logging.Level;
import java.util.logging.Logger;
import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentParser;
import dev.langchain4j.data.document.loader.FileSystemDocumentLoader;
import dev.langchain4j.data.document.parser.apache.tika.ApacheTikaDocumentParser;
import dev.langchain4j.data.document.DocumentSplitter;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.model.embedding.onnx.allminilml6v2.AllMiniLmL6V2EmbeddingModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.output.Response;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.service.AiServices;
import ma.emsi.KhalidYoussef.Assistant;

import java.util.List;
import java.util.Scanner;
public class Test2Logging {
    private static void configureLogger() {
        Logger packageLogger = Logger.getLogger("dev.langchain4j");
        packageLogger.setLevel(Level.FINE); // plus verbeux que INFO

        ConsoleHandler handler = new ConsoleHandler();
        handler.setLevel(Level.FINE);

        // Évite d'avoir les logs en double
        packageLogger.setUseParentHandlers(false);
        packageLogger.addHandler(handler);
    }
    public static void main(String[] args) throws URISyntaxException {
        configureLogger();

        String Key = System.getenv("GEMINI_KEY");

        ChatModel chatModel = GoogleAiGeminiChatModel.builder()
                .apiKey(Key)
                .modelName("gemini-2.5-flash")
                .temperature(0.3)
                .logRequestsAndResponses(true) // activation du logging HTTP côté LangChain4j
                .build();

        URL resource = RagNaif.class.getClassLoader().getResource("langchain4j.pdf");
        if (resource == null) {
            throw new IllegalStateException(
                    "Erreur : le fichier 'langchain4j.pdf' est introuvable.");
        }
        Path pdfPath = Paths.get(resource.toURI());
        // Lecture du PDF avec Apache Tika
        DocumentParser parser = new ApacheTikaDocumentParser();
        Document document = FileSystemDocumentLoader.loadDocument(pdfPath, parser);

        // Découpage du document en segments
        DocumentSplitter splitter = DocumentSplitters.recursive(300, 30);
        List<TextSegment> segments = splitter.split(document);

        // Génération des embeddings
        EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();
        Response<List<Embedding>> response = embeddingModel.embedAll(segments);
        List<Embedding> embeddings = response.content();

        // Stockage en mémoire
        EmbeddingStore<TextSegment> embeddingStore = new InMemoryEmbeddingStore<>();
        embeddingStore.addAll(embeddings, segments);

        System.out.println("✔ PHASE 1 terminée : "
                + segments.size() + " segments enregistrés dans le magasin d'embeddings.");


        // PHASE 2 : utilisation des embeddings pour répondre

        // Récupération des segments les plus pertinents (RAG)
        EmbeddingStoreContentRetriever contentRetriever =
                EmbeddingStoreContentRetriever.builder()
                        .embeddingStore(embeddingStore)
                        .embeddingModel(embeddingModel)
                        .maxResults(2)
                        .minScore(0.5)
                        .build();

        // Assistant avec mémoire (10 messages) + RAG
        Assistant assistant =
                AiServices.builder(Assistant.class)
                        .chatModel(chatModel)
                        .chatMemory(MessageWindowChatMemory.withMaxMessages(10))
                        .contentRetriever(contentRetriever)
                        .build();

        // Première question imposée par l’énoncé
        String questionInitiale = "Quelle est la définition du 'RAG' ; à quoi ça sert ?";
        String reponseInitiale = assistant.chat(questionInitiale);
        System.out.println("Question : " + questionInitiale);
        System.out.println("Réponse : " + reponseInitiale);

        try (Scanner scanner = new Scanner(System.in)) {
            while (true) {
                System.out.println("\n==================================================");
                System.out.println("Posez votre question: ");
                String question = scanner.nextLine();

                if ("bye".equalsIgnoreCase(question)) {
                    break;
                }
                if (question.isBlank()) {
                    continue;
                }

                String reponse = assistant.chat(question);
                System.out.println("--------------------------------------------------");
                System.out.println("Assistant : " + reponse);
                System.out.println("==================================================");
            }
        }
    }
}
