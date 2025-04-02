import torch
import torch.nn as nn
import torchvision.transforms.functional as TF


class DoubleConv(nn.Module):
    """
    DoubleConv is a class that defines a double convolutional layer with batch normalization and ReLU activation.
    It consists of two convolutional layers, each followed by batch normalization and ReLU activation.

    Attributes
    ----------
    conv : torch.nn.Sequential
        The double convolutional layer with batch normalization and ReLU activation.

    Parameters
    ----------
    nn : torch.nn.Module
        The base class for all neural network modules in PyTorch.

    """

    def __init__(self, in_channels, out_channels):
        """
        __init__ is the constructor method for the DoubleConv class.
        It initializes the double convolutional layer with batch normalization
        and ReLU activation.

        Parameters
        ----------
        in_channels : int
            The number of input channels for the first convolutional layer.
        out_channels : int
            The number of output channels for both convolutional layers.

        Returns
        -------
        None

        Examples
        --------
        >>> double_conv = DoubleConv(64, 128)
        >>> print(double_conv)
        DoubleConv(
          (conv): Sequential(
            (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU(inplace=True)
            (3): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (4): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (5): ReLU(inplace=True)
          )
        )

        """
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """
    UNet is a class that defines a U-Net architecture for image segmentation
    tasks.
    It consists of an encoder-decoder structure with skip connections.

    Parameters
    ----------
    nn : torch.nn.Module
        The base class for all neural network modules in PyTorch.
    """

    def __init__(
        self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]
    ):
        """
        __init__ is the constructor method for the UNet class.
        It initializes the U-Net architecture with encoder-decoder structure
        and skip connections.
        The encoder consists of several downsampling blocks, and the decoder
        consists of several upsampling blocks.
        Parameters
        ----------
        in_channels : int, optional
            input channels for the UNet encoder, by default 3
        out_channels : int, optional
            output channels for the UNet decoder, by default 1
        features : list, optional
            list of feature sizes for each layer in the UNet, by default [64, 128, 256, 512]

        Returns
        -------
        None

        Examples
        >>>x = torch.randn((3, 1, 161, 161))
        >>>model = UNet(in_channels=1, out_channels=1)
        >>>preds = model(x)
        >>> print(f"Input shape: {x.shape}")
        >>> print(f"Output shape: {preds.shape}")
        Input shape: torch.Size([3, 1, 161, 161])
        Output shape: torch.Size([3, 1, 161, 161])
        """
        super(UNet, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature * 2,
                    feature,
                    kernel_size=2,
                    stride=2,
                )
            )
            self.ups.append(DoubleConv(feature * 2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        """
        forward is the method that defines the forward pass of the U-Net.
        It takes an input tensor and passes it through the encoder,

        Parameters
        ----------
        x : tensor
            input tensor

        Returns
        -------
        x : tensor
            output from the final convolutional layer

        Examples
        --------
        >>> x = torch.randn((3, 1, 161, 161))
        >>> model = UNet(in_channels=1, out_channels=1)
        >>> preds = model(x)
        >>> print(f"Input shape: {x.shape}")
        >>> print(f"Output shape: {preds.shape}")
        >>> assert preds.shape == x.shape
        Input shape: torch.Size([3, 1, 161, 161])
        Output shape: torch.Size([3, 1, 161, 161])
        """
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]

            # Resize skip connection if needed
            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)

        return self.final_conv(x)

    @classmethod
    def plot_model(
        cls,
        in_channels=1,
        out_channels=1,
        input_size=(3, 1, 161, 161),
        show_shapes=True,
        save_graph=True,
        expand_nested=True,
        directory=".",
    ):
        """
        plot_model is a class method that plots the U-Net model architecture.
        It uses the torchview library to visualize the model graph. Requires
        torchview library to be installed and graphviz library to be installed.
        It saves the graph as a PNG image.

        Parameters
        ----------
        in_channels : int, optional
            number of input channels for the UNet encoder, by default 1
        out_channels : int, optional
            number of output channels for the UNet decoder, by default 1
        input_size : tuple, optional
            size of the input tensor, by default (3, 1, 161, 161)
        show_shapes : bool, optional
            whether to show the shapes of the tensors in the graph, by default True
        save_graph : bool, optional
            whether to save the graph as a PNG image, by default True
        expand_nested : bool, optional
            whether to expand nested modules in the graph, by default True
        directory : str, optional
            directory to save the graph image, by default "."

        Returns
        -------
        None

        Examples
        --------
        >>> UNet.plot_model(
        >>>     in_channels=1,
        >>>     out_channels=1,
        >>>     input_size=(3, 1, 161, 161),
        >>>     directory="model_plot/",
        >>> )
        >>> assert os.path.exists("model_plot/model.gv.png")
        """
        from torchview import draw_graph

        model = cls(in_channels=in_channels, out_channels=out_channels)

        # Generate the model visualization
        model_graph = draw_graph(
            model,
            input_size=input_size,
            show_shapes=show_shapes,
            save_graph=save_graph,
            expand_nested=expand_nested,
            directory=directory,
        )

        model_graph.visual_graph


if __name__ == "__main__":
    UNet.plot_model(
        in_channels=1,
        out_channels=1,
        input_size=(3, 1, 161, 161),
        directory="model_plot/",
    )
