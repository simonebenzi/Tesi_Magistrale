function [] = PlotNumberedClustersPosition(net)

%data_mean = net.X_mean;
%data_std  = net.X_std;

%border_x_min = data_mean(1) - 3*data_std(1);
%border_y_min = data_mean(2) - 3*data_std(2);
%border_x_max = data_mean(1) + 3*data_std(1);
%border_y_max = data_mean(2) + 3*data_std(2);

ColorFig = net.dataColorNode;
MeanFig  = net.nodesMean;
N        = size(net.nodesMean,1);

hold on
scatter(MeanFig(:,1),MeanFig(:,2),250,'.','MarkerEdgeColor','y',...
              'MarkerFaceColor','y')          % for the '+' at mean position of nodes
quiver(MeanFig(:,1),MeanFig(:,2),MeanFig(:,3),MeanFig(:,4),'LineWidth',1.8, ...
       'Color','r','MarkerEdgeColor', 'k','AutoScale','on', 'AutoScaleFactor', 2.4)
plot(MeanFig(:,1),MeanFig(:,2),'ko','MarkerFaceColor','y','MarkerEdgeColor','y','MarkerSize',16);
a = [0:N-1]'; b = num2str(a); c = cellstr(b);
     text(MeanFig(:,1), MeanFig(:,2), c, 'HorizontalAlignment', 'center', ...
     'VerticalAlignment', 'middle', 'FontSize',10);
grid on  

%axis([border_x_min border_x_max border_y_min border_y_max])
xlabel('x','FontSize',15, 'Interpreter','latex');
ylabel('y' ,'FontSize',15, 'Interpreter','latex');

end