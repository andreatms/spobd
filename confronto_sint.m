%% PULIZIA AMBIENTE
clc;
clear;
close all;


%% PARAMETRI
% Numero di cluster
clusters_num = 3;

% Numero di nodi per singolo cluster
nodes = [15, 25, 20];

% Probabilità di connessione dei nodi intra-cluster
p_intra = [0.5, 0.5, 0.5];

% Pesi degli archi intra-cluster
weights_intra = {[10, 25, 50, 100]};

% Pesi degli archi inter-cluster
weights_inter = {[1, 5, 10, 25, 50]};


% Definizione dei valori di p_inter da testare
p_inter_values = 0:0.05:1;
num_iterations = length(p_inter_values);
num_repeats = 100;


%% INIZIALIZZAZIONE
% Inizializzazione matrici per i risultati
results = struct();
results.p_inter_vals = p_inter_values;
results.accuracy_kcut = zeros(num_iterations, num_repeats);
results.accuracy_ratiocut = zeros(num_iterations, num_repeats);
results.accuracy_normcut = zeros(num_iterations, num_repeats);

% true labels per il calcolo della metrica
true_labels = create_true_labels(nodes);

fprintf('##### ANALISI ACCURATEZZA AL VARIARE DI p_inter #####\n\n');

% Ciclo for per iterazioni (variazione p_inter)
for iter = 1:num_iterations
    p_inter_val = p_inter_values(iter);
    p_inter = p_inter_val * ones(clusters_num, clusters_num);
    p_inter(eye(clusters_num) == 1) = 0;

    fprintf('Iterazione %d/%d: p_inter = %.2f\n', iter, num_iterations, p_inter_val);

    for rep = 1:num_repeats
        % Generazione del grafo
        A = graphN(clusters_num, nodes, p_intra, p_inter, weights_intra, weights_inter);

        % Uso del numero di cluster noto
        k = clusters_num;

        % Applicazione dei tre algoritmi
        clusters_kcut = spectral_k_cut(A, k);
        clusters_ratiocut = spectral_ratio_cut(A, k);
        clusters_normcut = spectral_k_norm_cut(A, k);

        % Calcolo dell'accuratezza per ogni metodo
        results.accuracy_kcut(iter, rep) = calculate_accuracy(true_labels, clusters_kcut);
        results.accuracy_ratiocut(iter, rep) = calculate_accuracy(true_labels, clusters_ratiocut);
        results.accuracy_normcut(iter, rep) = calculate_accuracy(true_labels, clusters_normcut);
    end

    fprintf('   Accuratezza media - K-Cut: %.2f%%, Ratio-Cut: %.2f%%, Norm-Cut: %.2f%%\n', ...
        mean(results.accuracy_kcut(iter, :))*100, ...
        mean(results.accuracy_ratiocut(iter, :))*100, ...
        mean(results.accuracy_normcut(iter, :))*100);
    fprintf('\n');
end

% Visualizzazione dei risultati
plot_accuracy_results(results);


%% FUNZIONI
% Memorizzazione etichette vere dei nodi
function true_labels = create_true_labels(nodes)
    % Crea il ground truth basato sulla struttura dei cluster
    true_labels = [];
    for i = 1:length(nodes)
        true_labels = [true_labels, i * ones(1, nodes(i))];
    end
end


% Calcola l'accuratezza della classificazione usando un approccio greedy
function accuracy = calculate_accuracy(true_labels, clusters)
    n_nodes = length(true_labels);
    predicted_labels = zeros(n_nodes, 1);
    
    for i = 1:length(clusters)
        predicted_labels(clusters{i}) = i;
    end
    
    k_true = max(true_labels);
    k_pred = length(clusters);
    
    % Matrice di confusione
    confusion_matrix = zeros(k_true, k_pred);
    for i = 1:n_nodes
        if predicted_labels(i) > 0
            confusion_matrix(true_labels(i), predicted_labels(i)) = ...
                confusion_matrix(true_labels(i), predicted_labels(i)) + 1;
        end
    end
    
    % Approccio greedy: assegna ogni cluster vero al cluster predetto con massima sovrapposizione
    used_pred = false(k_pred, 1);
    correct = 0;
    
    for i = 1:k_true
        best_overlap = 0;
        best_pred = 0;
        
        for j = 1:k_pred
            if ~used_pred(j) && confusion_matrix(i, j) > best_overlap
                best_overlap = confusion_matrix(i, j);
                best_pred = j;
            end
        end
        
        if best_pred > 0
            used_pred(best_pred) = true;
            correct = correct + best_overlap;
        end
    end
    
    accuracy = correct / n_nodes;
end


% Visualizzazione risultati della metrica utilizzata (media e deviazione standard)
function plot_accuracy_results(results)
    mean_kcut = mean(results.accuracy_kcut, 2) * 100;
    std_kcut = std(results.accuracy_kcut, 0, 2) * 100;
    mean_ratiocut = mean(results.accuracy_ratiocut, 2) * 100;
    std_ratiocut = std(results.accuracy_ratiocut, 0, 2) * 100;
    mean_normcut = mean(results.accuracy_normcut, 2) * 100;
    std_normcut = std(results.accuracy_normcut, 0, 2) * 100;

    figure('Position', [100, 100, 1000, 600]);

    % Grafico
    errorbar(results.p_inter_vals, mean_kcut, std_kcut, '-o', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', 'K-Cut');
    hold on;
    errorbar(results.p_inter_vals, mean_ratiocut, std_ratiocut, '-s', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', 'Ratio Cut');
    errorbar(results.p_inter_vals, mean_normcut, std_normcut, '-^', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', 'Normalized Cut');
    xlabel('p_{inter}', 'FontSize', 12);
    ylabel('Accuratezza (%)', 'FontSize', 12);
    title('Accuratezza di Classificazione vs p_{inter}', 'FontSize', 14);
    legend('Location', 'best', 'FontSize', 12);
    grid on;
    ylim([0, 100]);


    % Trova i valori ottimali per ogni metodo
    [best_acc_kcut, idx_kcut] = max(mean_kcut);
    [best_acc_ratiocut, idx_ratiocut] = max(mean_ratiocut);
    [best_acc_normcut, idx_normcut] = max(mean_normcut);

    % Trova il metodo migliore in assoluto
    all_accuracies = [best_acc_kcut, best_acc_ratiocut, best_acc_normcut];
    [overall_best, best_method_idx] = max(all_accuracies);
    method_names = {'K-Cut', 'Ratio Cut', 'Normalized Cut'};
    best_method = method_names{best_method_idx};
end


%% Funzioni di clustering spectral
% k-Cut
function clusters = spectral_k_cut(A, k)
    % Calcolo matrice D e L
    D = diag(sum(A, 2));
    L = D - A;
    
    % Calcolo autovettori di L
    opts.tol = 1e-6;
    opts.maxit = 1000;
    [U, ~] = eigs(L, k + 1, 'smallestreal', opts);
    
    % Esecuzione k-means e calcolo cluster
    labels = kmeans(U, k, 'Replicates', 10);
    clusters = cell(k, 1);
    for i = 1:k
        clusters{i} = find(labels == i);
    end
end

% Ratio Cut
function clusters = spectral_ratio_cut(A, k)
    % Calcolo matrice D e L
    D = diag(sum(A, 2));
    L = D - A;
    
    % Calcolo autovettori di L
    opts.tol = 1e-6;
    opts.maxit = 1000;
    [U, ~] = eigs(L, k + 1, 'smallestreal', opts);
    
    % Normalizzazione autovettori di L
    U_norm = U ./ (sqrt(sum(U.^2, 2)) + eps);
    
    % Esecuzione k-means e calcolo cluster
    labels = kmeans(U_norm, k, 'Replicates', 10);
    clusters = cell(k, 1);
    for i = 1:k
        clusters{i} = find(labels == i);
    end
end

% Normalized Cut
function clusters = spectral_k_norm_cut(A, k)
    % Calcolo matrice D e L_sym
    D = diag(sum(A, 2));
    D_inv_sqrt = diag(1 ./ sqrt(diag(D) + eps));
    L_sym = eye(size(A)) - D_inv_sqrt * A * D_inv_sqrt;

    % Calcolo autovettori di L_sym
    [U, ~] = eigs(L_sym, k, 'smallestreal');

    % Normalizzazione autovettori di L_sym
    U_norm = U ./ vecnorm(U, 2, 2);

    % Esecuzione k-means e calcolo cluster
    labels = kmeans(U_norm, k, 'Replicates', 10);
    clusters = cell(1, k);
    for i = 1:k
        clusters{i} = find(labels == i);
    end
end