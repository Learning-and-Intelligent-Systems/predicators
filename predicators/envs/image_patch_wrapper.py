from viper.image_patch import ImagePatch


class ImagePatchPlus(ImagePatch):

    def label_object(self, object: Object, label: str) -> ImagePatchPlus:
        """
        Label an object in the image patch.

        Parameters:
        -----------
        object : Object
            The object to be labeled.
        
        label : str
            The label to be assigned to the object.
        """
        pass

    def crop_to_objects(self, objects: List[Object]) -> ImagePatchPlus:
        """
        Crop the image patch to the smallest bounding box of the specified 
        objects.

        Parameters:
        -----------
        objects : List[Object]
            The objects whose bounding box is to be used for cropping.
        """
        pass